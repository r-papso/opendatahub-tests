"""Reverting external routing from Praxis back to OGX during the transition window.

The test drives the rollback through the OGXServer CR and verifies it from
outside the cluster, against the same Gateway hostname a client would use. The
hostname is resolved once, before the rollback, and reused afterwards, so a
rollback that only works because the client was reconfigured cannot pass.
"""

from collections.abc import Generator
from typing import Any

import httpx
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL, OGX_CORE_INFERENCE_MODEL, ModelInfo
from tests.ogx.praxis.constants import (
    NAMESPACE_PARAMS,
    OGX_SERVER_PARAMS,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSES_API_PATH,
)
from tests.ogx.praxis.upgrade.constants import (
    MAX_LISTED_RESOURCES,
    OGX_SERVER_ROLLBACK_PATCH,
    OGX_SERVICE_NAME_SUFFIX,
    ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
    ROLLBACK_POLL_INTERVAL,
    ROLLBACK_PROBE_MARKER,
    ROLLBACK_PROBE_MAX_OUTPUT_TOKENS,
    ROLLBACK_RESPONSES_TIMEOUT,
    SEED_CONVERSATIONS_COUNT,
    SEED_FILES_COUNT,
    SEED_MARKER,
    SEED_RESPONSES_COUNT,
)
from tests.ogx.praxis.upgrade.utils import (
    StateInventory,
    capture_state_inventory,
    first_successful_response,
    format_field_diff,
    load_baseline_section,
    save_baseline_section,
    seed_conversations,
    seed_files,
    seed_responses,
)
from tests.ogx.praxis.utils import (
    backend_services,
    gateway_base_url,
    http_routes_matching_path,
    pods_logging_marker,
    serving_pods_for_path,
)
from tests.ogx.utils import select_ogx_model
from utilities.exceptions import UnexpectedResourceCountError
from utilities.infra import get_openshift_token
from utilities.resources.http_route import HTTPRoute
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def responses_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """The single HTTPRoute that owns `POST /v1/responses` on the external Gateway."""
    http_routes = http_routes_matching_path(client=admin_client, path=RESPONSES_API_PATH)
    if not http_routes:
        pytest.skip(
            f"No HTTPRoute declares {RESPONSES_API_PATH}; the cluster does not expose the "
            "Responses API through the Gateway, so there is no external routing to roll back"
        )
    if len(http_routes) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 HTTPRoute declaring {RESPONSES_API_PATH}, found "
            f"{[f'{route.namespace}/{route.name}' for route in http_routes]}"
        )
    return http_routes[0]


@pytest.fixture(scope="class")
def external_responses_url(responses_http_route: HTTPRoute) -> str:
    """The external URL of `/v1/responses`, resolved once and never rebuilt."""
    return f"{gateway_base_url(http_route=responses_http_route)}{RESPONSES_API_PATH}"


@pytest.fixture(scope="class")
def praxis_serving_pods(
    admin_client: DynamicClient,
    responses_http_route: HTTPRoute,
    ogx_server: OgxServer,
) -> list[Pod]:
    """The pods serving `/v1/responses` before the rollback; skips unless they are Praxis."""
    backends = {str(service.name) for service in backend_services(client=admin_client, http_route=responses_http_route)}
    ogx_service_name = f"{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}"
    if ogx_service_name in backends:
        pytest.skip(
            f"HTTPRoute {responses_http_route.namespace}/{responses_http_route.name} already routes "
            f"{RESPONSES_API_PATH} to the OGX Service '{ogx_service_name}' (backends: {sorted(backends)}); "
            "the cluster is not in the post-migration state this test rolls back from"
        )
    serving_pods = serving_pods_for_path(client=admin_client, http_route=responses_http_route)
    if not serving_pods:
        pytest.skip(
            f"No pods back {RESPONSES_API_PATH} through Services {sorted(backends)}; "
            "the serving workload cannot be correlated against the request after the rollback"
        )
    LOGGER.info(f"{RESPONSES_API_PATH} is served by {[pod.name for pod in serving_pods]} before the rollback")
    return serving_pods


@pytest.fixture(scope="class")
def configured_rollback_patch() -> dict[str, Any]:
    """Skip unless a mechanism for reverting external routing to OGX has shipped."""
    if OGX_SERVER_ROLLBACK_PATCH is None:
        pytest.skip(
            "No rollback mechanism is defined: the strategy proposes a flag on the DataScienceCluster or the "
            "OGXServer CR but no such field has shipped. Set OGX_SERVER_ROLLBACK_PATCH in "
            "tests/ogx/praxis/upgrade/constants.py to the documented patch once it does"
        )
    return OGX_SERVER_ROLLBACK_PATCH


@pytest.fixture(scope="class")
def gateway_ogx_client(
    responses_http_route: HTTPRoute,
    gateway_authorization_header: dict[str, str],
) -> Generator[OgxClient]:
    """Client bound to the external Gateway hostname, built once and never reconfigured."""
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        yield OgxClient(
            base_url=gateway_base_url(http_route=responses_http_route),
            default_headers=gateway_authorization_header,
            http_client=http_client,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def gateway_authorization_header(admin_client: DynamicClient) -> dict[str, str]:
    """Authorization header sent with every request to the external Gateway hostname."""
    return {"Authorization": f"Bearer {get_openshift_token(client=admin_client)}"}


@pytest.fixture(scope="class")
def gateway_ogx_models(gateway_ogx_client: OgxClient) -> ModelInfo:
    """Models as the external Gateway hostname reports them."""
    return select_ogx_model(
        models=gateway_ogx_client.models.list().data,
        providers=gateway_ogx_client.providers.list(),
        configured_model=OGX_CORE_INFERENCE_MODEL,
    )


@pytest.fixture(scope="class")
def sampled_state_ids(gateway_ogx_client: OgxClient, gateway_ogx_models: ModelInfo) -> dict[str, list[str]]:
    """Files, vector stores, conversations and responses created before the rollback.

    State is seeded rather than discovered so that the inventory comparison has
    something to lose; comparing empty counts would pass whatever the rollback did.
    """
    conversation_ids = seed_conversations(ogx_client=gateway_ogx_client, count=SEED_CONVERSATIONS_COUNT)
    # The vector store is left empty on purpose: an attached file would keep its
    # status moving while it is ingested, and status is a compared field.
    vector_store = gateway_ogx_client.vector_stores.create(name=f"{SEED_MARKER}-rollback")
    return {
        "files": seed_files(ogx_client=gateway_ogx_client, count=SEED_FILES_COUNT),
        "vector_stores": [vector_store.id],
        "responses": seed_responses(
            ogx_client=gateway_ogx_client,
            model_id=gateway_ogx_models.model_id,
            count=SEED_RESPONSES_COUNT,
            conversation_id=conversation_ids[0],
        ),
        "conversations": conversation_ids,
    }


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
@pytest.mark.slow
class TestRollbackFromPraxisToOgx:
    """Rolling external routing back from Praxis to OGX during the transition window."""

    @pytest.mark.post_upgrade
    def test_rollback_from_praxis_to_ogx(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_server: OgxServer,
        responses_http_route: HTTPRoute,
        external_responses_url: str,
        praxis_serving_pods: list[Pod],
        configured_rollback_patch: dict[str, Any],
        gateway_ogx_client: OgxClient,
        gateway_authorization_header: dict[str, str],
        gateway_ogx_models: ModelInfo,
        sampled_state_ids: dict[str, list[str]],
    ) -> None:
        """Verify that external routing can be reverted from Praxis to OGX without losing state.

        Given: An upgraded cluster whose external Gateway hostname routes
            `/v1/responses` to Praxis, holding files, vector stores,
            conversations and responses.
        When: The documented rollback procedure is applied and the unchanged
            external hostname is polled every 15 seconds.
        Then: `POST /v1/responses` answers HTTP 200 within five minutes, the
            path is served by the OGX Service again, Praxis did not serve the
            request, and every count and sampled resource is unchanged.
        """
        inventory_before = capture_state_inventory(ogx_client=gateway_ogx_client, sampled_ids=sampled_state_ids)
        assert not inventory_before["missing"], (
            f"State sampled before the rollback does not resolve: {inventory_before['missing']}"
        )
        # Persisted through the shared baseline mechanism so the T0 snapshot
        # outlives the rollback, which restarts the serving workload, and stays
        # inspectable on the cluster when the run fails part way through.
        save_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
            payload=inventory_before,
        )

        ResourceEditor(patches={ogx_server: configured_rollback_patch}).update()
        LOGGER.info(f"Applied rollback patch {configured_rollback_patch} to OGXServer {ogx_server.name}")

        # The five-minute bound is the poll window itself: the sampler gives up,
        # and the test fails, as soon as it is exceeded without an HTTP 200.
        probe_body, elapsed_seconds = first_successful_response(
            url=external_responses_url,
            headers={**gateway_authorization_header, "Content-Type": "application/json"},
            payload={
                "model": gateway_ogx_models.model_id,
                "input": f"Reply with the single word '{ROLLBACK_PROBE_MARKER}'.",
                "store": False,
                "stream": False,
                "max_output_tokens": ROLLBACK_PROBE_MAX_OUTPUT_TOKENS,
            },
            timeout=ROLLBACK_RESPONSES_TIMEOUT,
            interval=ROLLBACK_POLL_INTERVAL,
        )
        LOGGER.info(f"First HTTP 200 after the rollback arrived {elapsed_seconds:.1f}s after T0")

        rolled_back_backends = {
            str(service.name) for service in backend_services(client=admin_client, http_route=responses_http_route)
        }
        assert rolled_back_backends == {f"{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}"}, (
            f"After the rollback, {RESPONSES_API_PATH} should be backed by the OGX Service "
            f"'{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}' alone, but HTTPRoute "
            f"{responses_http_route.namespace}/{responses_http_route.name} forwards to {sorted(rolled_back_backends)}"
        )

        probe_id = str(probe_body["id"])
        assert not pods_logging_marker(pods=praxis_serving_pods, marker=probe_id), (
            f"Response '{probe_id}' appears in the logs of the pods that served {RESPONSES_API_PATH} before the "
            "rollback, so Praxis served the request that was expected to reach OGX"
        )
        ogx_pods_logging_probe = pods_logging_marker(
            pods=serving_pods_for_path(client=admin_client, http_route=responses_http_route),
            marker=probe_id,
        )
        if ogx_pods_logging_probe:
            LOGGER.info(f"Response '{probe_id}' was logged by OGX pods {ogx_pods_logging_probe}")
        else:
            # The negative half above is the binding check. Whether OGX logs the
            # response id at all depends on the distribution's log level, so its
            # absence is reported rather than failed.
            LOGGER.warning(
                f"Response '{probe_id}' was not found in the logs of the OGX pods now backing "
                f"{RESPONSES_API_PATH}; positive log correlation is inconclusive for this run"
            )

        inventory_after = capture_state_inventory(ogx_client=gateway_ogx_client, sampled_ids=sampled_state_ids)
        recorded_inventory: StateInventory = load_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
        )

        assert not inventory_after["missing"], (
            f"Sampled state no longer resolves after the rollback: {inventory_after['missing']}"
        )
        assert inventory_after["counts"] == recorded_inventory["counts"], (
            f"State counts changed across the rollback: {recorded_inventory['counts']} -> "
            f"{inventory_after['counts']} (listings are capped at {MAX_LISTED_RESOURCES} items)"
        )

        differences = [
            difference
            for kind, sampled_before in recorded_inventory["sampled"].items()
            for resource_id, fields_before in sampled_before.items()
            if (
                difference := format_field_diff(
                    resource=kind,
                    resource_id=resource_id,
                    before=fields_before,
                    after=inventory_after["sampled"][kind][resource_id],
                )
            )
        ]
        assert not differences, "Sampled resources changed across the rollback: " + "; ".join(differences)

        assert external_responses_url.startswith(gateway_base_url(http_route=responses_http_route)), (
            f"The external hostname changed during the rollback: the client kept using {external_responses_url} "
            f"while the Gateway now exposes {gateway_base_url(http_route=responses_http_route)}"
        )
