"""Delegation error-path tests for Praxis fronting OGX.

Objective: when Praxis cannot reach OGX it must return a structured OpenAI-compatible
server error rather than hanging or surfacing a platform-internal error.

These tests are destructive: OGX is scaled to zero replicas to sever the delegation path.
Restoration happens in fixture teardown, so a failing test cannot leave the cluster broken.
"""

from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.ogx.praxis.constants import (
    NAMESPACE_PARAMS,
    PRAXIS_MODE_OGX_SERVER_PARAMS,
    REQUEST_TIMEOUT_SECONDS,
)
from utilities.constants import Timeout
from utilities.exceptions import ResourceNotReadyError
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Model used only to trigger the delegation path; the response content is never asserted
# on, because the request is expected to fail before inference happens.
PROBE_MODEL: str = "praxis-delegation-error-probe"


def is_server_error(response: requests.Response) -> bool:
    """Whether a response carries an HTTP 5xx status.

    Args:
        response: Response returned at the public Gateway boundary.

    Returns:
        True when the status code is in the 5xx range.
    """
    return 500 <= response.status_code < 600


def ogx_service_has_endpoints(client: DynamicClient, ogx_server: OgxServer) -> bool:
    """Report whether the OGX service still has ready endpoint addresses.

    Args:
        client: Client with read access to the OGX namespace.
        ogx_server: Server whose backing pods are inspected.

    Returns:
        True while at least one OGX pod is still running.
    """
    return bool(
        list(
            Pod.get(
                dyn_client=client,
                namespace=ogx_server.namespace,
                label_selector=f"app={ogx_server.name}",
            )
        )
    )


def assert_openai_error_envelope(response: requests.Response) -> None:
    """Assert a response body matches the OpenAI-compatible error schema.

    Args:
        response: Error response returned at the public Gateway boundary.

    Raises:
        AssertionError: If the body is not JSON, or its `error` object is missing any of
            the `type`, `message` and `code` fields, or any of them is empty.
    """
    try:
        payload = response.json()
    except ValueError as json_error:
        raise AssertionError(f"Expected an OpenAI-compatible JSON error body, got {response.text!r}") from json_error

    error = payload.get("error")
    assert isinstance(error, dict), f"Expected an 'error' object in the body, got {payload!r}"
    for field in ("type", "message", "code"):
        value = error.get(field)
        assert isinstance(value, str) and value.strip(), (
            f"Expected a non-empty 'error.{field}' in the OpenAI-compatible error body, got {error!r}"
        )


@pytest.fixture(scope="class")
def unreachable_ogx(
    unprivileged_client: DynamicClient,
    ogx_server: OgxServer,
) -> Generator[OgxServer, Any, Any]:
    """OGX scaled to zero replicas, restored on teardown even when a test fails."""
    ogx_deployment = Deployment(
        client=unprivileged_client,
        name=ogx_server.name,
        namespace=ogx_server.namespace,
        ensure_exists=True,
    )
    original_replicas = int(ogx_deployment.instance.spec.replicas or 1)
    ogx_deployment.scale_replicas(replica_count=0)
    try:
        for has_endpoints in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_4MIN,
            sleep=5,
            func=ogx_service_has_endpoints,
            client=unprivileged_client,
            ogx_server=ogx_server,
        ):
            if not has_endpoints:
                break
        else:
            raise ResourceNotReadyError(
                f"OGX pods for {ogx_server.namespace}/{ogx_server.name} did not terminate, "
                "so the delegation path was never severed"
            )
        LOGGER.info(f"OGX {ogx_server.name} scaled to zero; delegation path severed")
        yield ogx_server
    finally:
        ogx_deployment.scale_replicas(replica_count=original_replicas)
        ogx_deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_10MIN)
        LOGGER.info(f"OGX {ogx_server.name} restored to {original_replicas} replica(s)")


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {**NAMESPACE_PARAMS, "randomize_name": True},
            PRAXIS_MODE_OGX_SERVER_PARAMS,
            id="test_praxis_delegation_errors",
        )
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier3
@pytest.mark.slow
class TestPraxisErrorWhenOgxUnreachable:
    """Praxis reports an OpenAI-compatible 5xx error when OGX is unreachable."""

    def test_files_request_returns_openai_compatible_error(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_files_url: str,
        unreachable_ogx: OgxServer,
    ) -> None:
        """TC-NEG-002: The Files API fails cleanly when OGX is unreachable.

        Given: OGX is scaled to zero, so Praxis cannot delegate file state.
        When: The Files API is called through the public Praxis boundary.
        Then: Praxis returns an HTTP 5xx with an OpenAI-compatible error body.
        """
        response = request_session.get(
            url=praxis_files_url,
            headers=tenant_authorization_header,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        assert is_server_error(response=response), (
            f"Expected an HTTP 5xx from {praxis_files_url} while OGX is unreachable, got "
            f"HTTP {response.status_code}: {response.text}"
        )
        assert_openai_error_envelope(response=response)

    def test_responses_request_returns_openai_compatible_error(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_responses_url: str,
        unreachable_ogx: OgxServer,
    ) -> None:
        """TC-NEG-002: The Responses API fails cleanly when OGX is unreachable.

        Given: OGX is scaled to zero, so Praxis cannot delegate the request.
        When: The Responses API is called through the public Praxis boundary.
        Then: Praxis returns an HTTP 5xx with an OpenAI-compatible error body.
        """
        response = request_session.post(
            url=praxis_responses_url,
            headers=tenant_authorization_header,
            json={"model": PROBE_MODEL, "input": "ping"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        assert is_server_error(response=response), (
            f"Expected an HTTP 5xx from {praxis_responses_url} while OGX is unreachable, got "
            f"HTTP {response.status_code}: {response.text}"
        )
        assert_openai_error_envelope(response=response)
