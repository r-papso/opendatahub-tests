"""Reverting external routing from Praxis back to OGX during the transition window.

The rollback itself is not driven from here: like the upgrade it brackets, it is
applied to the test environment between the two pytest runs, so the coverage is
split into a pre-rollback test that records what must survive and a post-rollback
test that checks it did.

Both runs reach the server the same way a client would, through the external
Gateway hostname. The hostname is recorded before the rollback and compared
afterwards, so a rollback that only works because the client was reconfigured
cannot pass.
"""

from collections.abc import Generator

import httpx
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL, ModelInfo
from tests.ogx.praxis.constants import (
    NAMESPACE_PARAMS,
    OGX_SERVER_PARAMS,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSES_API_PATH,
)
from tests.ogx.praxis.upgrade.constants import (
    MAX_LISTED_RESOURCES,
    OGX_SERVICE_NAME_SUFFIX,
    ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
    ROLLBACK_PROBE_MARKER,
    SEED_CONVERSATIONS_COUNT,
    SEED_FILES_COUNT,
    SEED_MARKER,
    SEED_RESPONSES_COUNT,
)
from tests.ogx.praxis.upgrade.utils import (
    RollbackBaseline,
    capture_state_inventory,
    format_field_diff,
    load_baseline_section,
    sampled_resource_ids,
    save_baseline_section,
    seed_conversations,
    seed_files,
    seed_responses,
)
from tests.ogx.praxis.utils import (
    gateway_base_url,
    http_routes_matching_path,
    pods_for_service,
    pods_logging_marker,
)
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
    """The external URL of `/v1/responses`, as a client outside the cluster addresses it."""
    return f"{gateway_base_url(http_route=responses_http_route)}{RESPONSES_API_PATH}"


@pytest.fixture(scope="class")
def gateway_ogx_client(admin_client: DynamicClient, responses_http_route: HTTPRoute) -> Generator[OgxClient]:
    """Client bound to the external Gateway hostname, not to the OGX Route.

    The shared `ogx_client` addresses the OGX Service directly, so it would reach
    OGX whatever the external routing does; this test needs the hostname whose
    routing the rollback changes.
    """
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        yield OgxClient(
            base_url=gateway_base_url(http_route=responses_http_route),
            default_headers={"Authorization": f"Bearer {get_openshift_token(client=admin_client)}"},
            http_client=http_client,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def ogx_serving_pods(unprivileged_client: DynamicClient, ogx_server: OgxServer) -> list[Pod]:
    """The pods behind the OGX Service, whose logs must show the post-rollback request.

    Raises:
        ResourceNotFoundError: If the OGX Service selects no pod, leaving nothing
            to correlate the request against.
    """
    service = Service(
        client=unprivileged_client,
        name=f"{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}",
        namespace=ogx_server.namespace,
        ensure_exists=True,
    )
    pods = pods_for_service(client=unprivileged_client, service=service)
    if not pods:
        raise ResourceNotFoundError(f"Service {service.namespace}/{service.name} selects no OGX pod")
    LOGGER.info(f"OGX is served by {[pod.name for pod in pods]}")
    return pods


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
@pytest.mark.slow
class TestPreRollbackFromPraxisToOgx:
    """Recording what the rollback from Praxis back to OGX must preserve."""

    @pytest.mark.pre_upgrade
    def test_capture_state_inventory_pre_rollback(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_server: OgxServer,
        external_responses_url: str,
        gateway_ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """Capture the state inventory the rollback has to leave intact.

        Given: A cluster whose external Gateway hostname serves `/v1/responses`.
        When: Files, vector stores, conversations and responses are created
            through it, then counted and read back.
        Then: Every created resource resolves, and the inventory and the external
            URL are persisted for the post-rollback run to compare against.
        """
        # State is seeded rather than discovered so that the inventory comparison
        # has something to lose; comparing empty counts would pass whatever the
        # rollback did.
        conversation_ids = seed_conversations(ogx_client=gateway_ogx_client, count=SEED_CONVERSATIONS_COUNT)
        # The vector store is left empty on purpose: an attached file would keep
        # its status moving while it is ingested, and status is a compared field.
        vector_store = gateway_ogx_client.vector_stores.create(name=f"{SEED_MARKER}-rollback")
        sampled_state_ids = {
            "files": seed_files(ogx_client=gateway_ogx_client, count=SEED_FILES_COUNT),
            "vector_stores": [vector_store.id],
            "responses": seed_responses(
                ogx_client=gateway_ogx_client,
                model_id=ogx_models.model_id,
                count=SEED_RESPONSES_COUNT,
                conversation_id=conversation_ids[0],
            ),
            "conversations": conversation_ids,
        }

        inventory = capture_state_inventory(ogx_client=gateway_ogx_client, sampled_ids=sampled_state_ids)
        assert not inventory["missing"], f"State sampled before the rollback does not resolve: {inventory['missing']}"

        save_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
            payload=RollbackBaseline(inventory=inventory, external_url=external_responses_url),
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
@pytest.mark.slow
class TestPostRollbackFromPraxisToOgx:
    """Rolling external routing back from Praxis to OGX during the transition window."""

    @pytest.mark.post_upgrade
    def test_rollback_serves_from_ogx_with_state_intact(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_server: OgxServer,
        external_responses_url: str,
        ogx_serving_pods: list[Pod],
        gateway_ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """Verify that OGX serves the rolled-back endpoint and that no state was lost.

        Given: A cluster that routed `/v1/responses` to Praxis and has since had
            the documented rollback procedure applied to it.
        When: A response is created through the unchanged external hostname and
            the pre-rollback state inventory is re-captured through it.
        Then: The new response appears in the OGX pod logs, so OGX and not Praxis
            served it, and every count and sampled resource is unchanged.
        """
        baseline: RollbackBaseline = load_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=ROLLBACK_INVENTORY_CONFIG_MAP_KEY,
        )
        assert external_responses_url == baseline["external_url"], (
            f"The external hostname changed across the rollback: it was {baseline['external_url']} before and is "
            f"{external_responses_url} now, so the client would have had to be reconfigured"
        )

        # Unstored on purpose, so that probing does not change the response count
        # the inventory compares.
        probe = gateway_ogx_client.responses.create(
            input=f"Reply with the single word '{ROLLBACK_PROBE_MARKER}'.",
            model=ogx_models.model_id,
            store=False,
            stream=False,
        )
        # Correlating by response id requires the distribution to log it; a quiet
        # log level makes this fail rather than pass silently, which is the point:
        # without the correlation there is no evidence of which workload answered.
        pods_serving_probe = pods_logging_marker(pods=ogx_serving_pods, marker=probe.id)
        assert pods_serving_probe, (
            f"Response '{probe.id}' created through {external_responses_url} does not appear in the logs of the OGX "
            f"pods {[pod.name for pod in ogx_serving_pods]}, so there is no evidence OGX rather than Praxis served it"
        )
        LOGGER.info(f"Response '{probe.id}' was served by OGX pods {pods_serving_probe}")

        recorded_inventory = baseline["inventory"]
        inventory_after = capture_state_inventory(
            ogx_client=gateway_ogx_client,
            sampled_ids=sampled_resource_ids(inventory=recorded_inventory),
        )

        assert not inventory_after["missing"], (
            f"Sampled state no longer resolves through the rolled-back endpoint: {inventory_after['missing']}"
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
