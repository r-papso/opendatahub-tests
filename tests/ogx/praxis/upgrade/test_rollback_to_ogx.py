"""Rollback to OGX after an isolated OGX -> Praxis migration.

The module owns the complete lifecycle: it seeds OGX, runs the Praxis migration,
captures the migrated state, rolls external routing back to OGX, and verifies
that the unchanged Gateway hostname serves the same state from OGX.
"""

from collections.abc import Callable
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from ogx_client import OgxClient
from timeout_sampler import TimeoutSampler

from tests.ogx.constants import ModelInfo
from tests.ogx.praxis.constants import RESPONSES_API_PATH
from tests.ogx.praxis.upgrade.constants import (
    MAX_LISTED_RESOURCES,
    OGX_SERVICE_NAME_SUFFIX,
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
    sampled_resource_ids,
    seed_conversations,
    seed_files,
    seed_responses,
)
from tests.ogx.praxis.utils import backend_services, gateway_base_url, pods_for_service, pods_logging_marker
from utilities.resources.http_route import HTTPRoute
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# This migration owns a namespace and OGXServer so it cannot share state with
# another OGX -> Praxis migration module.
NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-praxis-rollback"}
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
}

OGX_SERVER_ROLLBACK_PATCH: dict[str, Any] = {"spec": {"praxisMode": {"enabled": False}}}
ROLLBACK_RECONCILIATION_TIMEOUT: int = 300
ROLLBACK_POLL_INTERVAL: int = 5


def wait_for_responses_route_to_ogx(
    admin_client: DynamicClient,
    ogx_server: OgxServer,
    responses_http_route: HTTPRoute,
) -> set[str]:
    """Wait until the Responses API route is backed only by the OGX Service.

    Args:
        admin_client: Client with cluster-wide access to resolve route backends.
        ogx_server: OGXServer whose Service must become the route backend.
        responses_http_route: Public HTTPRoute serving the Responses API.

    Returns:
        Names of the Services backing the Responses API after reconciliation.
    """
    expected_backend = f"{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}"
    rolled_back_backends: set[str] = set()
    for services in TimeoutSampler(
        wait_timeout=ROLLBACK_RECONCILIATION_TIMEOUT,
        sleep=ROLLBACK_POLL_INTERVAL,
        func=backend_services,
        client=admin_client,
        http_route=responses_http_route,
        path=RESPONSES_API_PATH,
    ):
        rolled_back_backends = {str(service.name) for service in services}
        if rolled_back_backends == {expected_backend}:
            break

    return rolled_back_backends


def rollback_to_ogx(
    admin_client: DynamicClient,
    ogx_server: OgxServer,
    responses_http_route: HTTPRoute,
) -> set[str]:
    """Disable Praxis mode and wait for the Responses route to return to OGX.

    Args:
        admin_client: Client with cluster-wide access to resolve route backends.
        ogx_server: OGXServer whose Praxis mode is disabled.
        responses_http_route: Public HTTPRoute serving the Responses API.

    Returns:
        Names of the Services backing the Responses API after reconciliation.
    """
    ResourceEditor(patches={ogx_server: OGX_SERVER_ROLLBACK_PATCH}).update()
    LOGGER.info(f"Applied rollback patch to OGXServer {ogx_server.namespace}/{ogx_server.name}")

    return wait_for_responses_route_to_ogx(
        admin_client=admin_client,
        ogx_server=ogx_server,
        responses_http_route=responses_http_route,
    )


@pytest.fixture(scope="class")
def ogx_serving_pods(unprivileged_client: DynamicClient, ogx_server: OgxServer) -> list[Pod]:
    """The pods behind the OGX Service after rollback."""
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
class TestRollbackFromPraxisToOgx:
    """Verify an isolated migration can be rolled back to OGX."""

    def test_rollback_serves_from_ogx_with_state_intact(
        self,
        admin_client: DynamicClient,
        ogx_server: OgxServer,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        praxis_migration_runner: Callable[[], OgxServer],
        praxis_client: OgxClient,
        praxis_responses_url: str,
        praxis_responses_http_route: HTTPRoute,
        ogx_serving_pods: list[Pod],
    ) -> None:
        """Verify OGX serves the rolled-back endpoint without losing migrated state.

        Given: An isolated OGXServer with files, a vector store, conversations,
            and responses seeded before migration to Praxis.
        When: Praxis migration is run and then `praxisMode` is disabled to roll
            the external endpoint back to OGX.
        Then: The unchanged Gateway hostname is served by OGX, and every count
            and sampled resource is unchanged from immediately before rollback.
        """
        conversation_ids = seed_conversations(ogx_client=ogx_client, count=SEED_CONVERSATIONS_COUNT)
        # The vector store is left empty so its compared status cannot change
        # asynchronously while a file is ingested.
        vector_store = ogx_client.vector_stores.create(name=f"{SEED_MARKER}-rollback")
        seeded_state_ids = {
            "files": seed_files(ogx_client=ogx_client, count=SEED_FILES_COUNT),
            "vector_stores": [vector_store.id],
            "responses": seed_responses(
                ogx_client=ogx_client,
                model_id=ogx_models.model_id,
                count=SEED_RESPONSES_COUNT,
                conversation_id=conversation_ids[0],
            ),
            "conversations": conversation_ids,
        }

        praxis_migration_runner()
        LOGGER.info("Completed OGX to Praxis migration")

        inventory_before = capture_state_inventory(ogx_client=praxis_client, sampled_ids=seeded_state_ids)
        assert not inventory_before["missing"], (
            f"Migrated state sampled before rollback does not resolve: {inventory_before['missing']}"
        )
        baseline = RollbackBaseline(inventory=inventory_before, external_url=praxis_responses_url)

        expected_backend = f"{ogx_server.name}{OGX_SERVICE_NAME_SUFFIX}"
        rolled_back_backends = rollback_to_ogx(
            admin_client=admin_client,
            ogx_server=ogx_server,
            responses_http_route=praxis_responses_http_route,
        )

        assert rolled_back_backends == {expected_backend}, (
            f"After rollback, {RESPONSES_API_PATH} should be backed by OGX Service '{expected_backend}' alone, "
            f"but HTTPRoute {praxis_responses_http_route.namespace}/{praxis_responses_http_route.name} forwards to "
            f"{sorted(rolled_back_backends)}"
        )

        current_responses_url = f"{gateway_base_url(http_route=praxis_responses_http_route)}{RESPONSES_API_PATH}"
        assert current_responses_url == baseline["external_url"], (
            f"The external hostname changed across rollback: it was {baseline['external_url']} before and is "
            f"{current_responses_url} now, so the client would have had to be reconfigured"
        )

        # Unstored on purpose, so probing does not change the response count.
        probe = praxis_client.responses.create(
            input=f"Reply with the single word '{ROLLBACK_PROBE_MARKER}'.",
            model=ogx_models.model_id,
            store=False,
            stream=False,
        )
        pods_serving_probe = pods_logging_marker(pods=ogx_serving_pods, marker=probe.id)
        assert pods_serving_probe, (
            f"Response '{probe.id}' created through {current_responses_url} does not appear in the logs of the OGX "
            f"pods {[pod.name for pod in ogx_serving_pods]}, so there is no evidence OGX rather than Praxis served it"
        )
        LOGGER.info(f"Response '{probe.id}' was served by OGX pods {pods_serving_probe}")

        recorded_inventory = baseline["inventory"]
        inventory_after = capture_state_inventory(
            ogx_client=praxis_client,
            sampled_ids=sampled_resource_ids(inventory=recorded_inventory),
        )

        assert not inventory_after["missing"], (
            f"Sampled state no longer resolves through the rolled-back endpoint: {inventory_after['missing']}"
        )
        assert inventory_after["counts"] == recorded_inventory["counts"], (
            f"State counts changed across rollback: {recorded_inventory['counts']} -> "
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
        assert not differences, "Sampled resources changed across rollback: " + "; ".join(differences)
