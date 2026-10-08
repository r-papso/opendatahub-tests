"""Delegation-boundary tests for OGX fronted by Praxis.

Objective: the OGX delegation port (8321) is not reachable from outside the cluster, and
Praxis rejects unauthenticated requests at the public boundary.
"""

from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.ingress_networking_k8s_io import Ingress
from ocp_resources.network_policy import NetworkPolicy

from tests.ogx.praxis.constants import (
    NAMESPACE_PARAMS,
    OGX_DELEGATION_PORT,
    PRAXIS_MODE_OGX_SERVER_PARAMS,
    REQUEST_TIMEOUT_SECONDS,
)
from utilities.constants import Timeout
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Label identifying the OpenShift router namespaces; it must never appear as an OGX peer.
ROUTER_POLICY_GROUP_LABEL: str = "network.openshift.io/policy-group"

MALFORMED_BEARER_TOKEN: str = "Bearer not.a.valid.token"


def ingress_peers_on_port(network_policy: NetworkPolicy, port: int) -> list[dict[str, Any]]:
    """Return the ingress peers admitted on a numeric port, across all rules.

    Args:
        network_policy: Policy whose ``spec.ingress`` rules are inspected.
        port: Numeric port the rule must admit.

    Returns:
        The flattened list of peers of every rule declaring ``port`` among its ports.
    """
    return [
        peer
        for rule in network_policy.instance.to_dict()["spec"].get("ingress") or []
        if any(rule_port.get("port") == port for rule_port in rule.get("ports") or [])
        for peer in rule.get("from") or []
    ]


def is_openshift_router_peer(peer: dict[str, Any]) -> bool:
    """Whether an ingress peer admits the OpenShift router namespaces.

    Args:
        peer: A single ``spec.ingress[].from[]`` entry.

    Returns:
        True when the peer selects namespaces labelled as the ingress policy group.
    """
    namespace_selector = peer.get("namespaceSelector") or {}
    return (namespace_selector.get("matchLabels") or {}).get(ROUTER_POLICY_GROUP_LABEL) == "ingress"


@pytest.fixture(scope="class")
def ogx_network_policy(
    unprivileged_client: DynamicClient,
    ogx_server: OgxServer,
) -> NetworkPolicy:
    """Operator-managed NetworkPolicy guarding the OGX delegation port."""
    network_policy = NetworkPolicy(
        client=unprivileged_client,
        name=f"{ogx_server.name}-network-policy",
        namespace=ogx_server.namespace,
    )
    network_policy.wait(timeout=Timeout.TIMEOUT_2MIN)
    return network_policy


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {**NAMESPACE_PARAMS, "randomize_name": True},
            PRAXIS_MODE_OGX_SERVER_PARAMS,
            id="test_praxis_delegation_boundary",
        )
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier3
class TestOgxDelegationPortBoundary:
    """The OGX delegation port is not externally reachable, and Praxis rejects unauthenticated callers."""

    def test_ogx_delegation_port_not_admitted_from_openshift_router(
        self,
        ogx_network_policy: NetworkPolicy,
    ) -> None:
        """No NetworkPolicy peer admits the OpenShift router on the delegation port.

        Given: An OGXServer deployed in Praxis-fronted internal-only mode.
        When: The operator-managed NetworkPolicy guarding the delegation port is inspected.
        Then: No peer on that port selects the OpenShift router namespaces, so off-cluster traffic
            arriving through the router cannot reach the delegation port.
        """
        router_peers = [
            peer
            for peer in ingress_peers_on_port(network_policy=ogx_network_policy, port=OGX_DELEGATION_PORT)
            if is_openshift_router_peer(peer=peer)
        ]
        assert not router_peers, (
            f"NetworkPolicy {ogx_network_policy.namespace}/{ogx_network_policy.name} admits the OpenShift router "
            f"on port {OGX_DELEGATION_PORT} through {router_peers}, which would expose the delegation port "
            "outside the cluster"
        )

    def test_ogx_has_no_external_ingress_exposure(
        self,
        unprivileged_client: DynamicClient,
        ogx_server: OgxServer,
    ) -> None:
        """No Ingress is created for OGX even when external access is requested.

        Given: An OGXServer in Praxis-fronted mode that explicitly sets
            spec.network.externalAccess.enabled to true.
        When: The namespace is searched for the Ingress the operator would create in legacy mode.
        Then: No such Ingress exists, so the request is not honoured and OGX stays internal-only.
        """
        requested_external_access = (ogx_server.instance.to_dict()["spec"].get("network") or {}).get(
            "externalAccess"
        ) or {}
        assert requested_external_access.get("enabled") is True, (
            "The OGXServer must request spec.network.externalAccess.enabled=true so this test proves the "
            f"setting is ignored rather than merely unset; got {requested_external_access}"
        )

        ingress = Ingress(
            client=unprivileged_client,
            name=f"{ogx_server.name}-ingress",
            namespace=ogx_server.namespace,
        )
        assert not ingress.exists, (
            f"Ingress {ingress.namespace}/{ingress.name} exists although OGX is internal-only; external access "
            "was requested and must not be honoured in Praxis-fronted mode"
        )

    def test_ogx_status_reports_internal_only_endpoint(self, ogx_server: OgxServer) -> None:
        """The OGXServer status advertises an internal endpoint and no external URL.

        Given: An OGXServer in Praxis-fronted mode that requested external access.
        When: Its status is read.
        Then: status.serviceURL is the internal cluster DNS endpoint and status.externalURL is empty.
        """
        status = ogx_server.instance.to_dict().get("status") or {}
        service_url = str(status.get("serviceURL") or "")
        assert service_url, (
            f"OGXServer {ogx_server.namespace}/{ogx_server.name} reports no status.serviceURL; status={status}"
        )
        assert ".svc.cluster.local" in service_url, (
            f"status.serviceURL {service_url!r} is not an internal cluster DNS endpoint"
        )
        assert not status.get("externalURL"), (
            f"status.externalURL is {status.get('externalURL')!r}; it must be empty for internal-only OGX"
        )

    def test_unauthenticated_responses_request_is_rejected(
        self,
        request_session: requests.Session,
        praxis_responses_url: str,
        tenant_authorization_header: dict[str, str],
        ogx_server: OgxServer,
    ) -> None:
        """Unauthenticated calls to the public Responses API are rejected.

        Given: Praxis serves the Responses API at the public Gateway boundary and an
            authenticated tenant is accepted there.
        When: The same request is sent with no Authorization header.
        Then: The response is HTTP 401.

        The authenticated call is the control: without it, a boundary that rejects every caller
        would look like correct authentication enforcement.
        """
        payload = {"model": "unauthenticated-probe", "input": "ping"}

        authenticated_response = request_session.post(
            url=praxis_responses_url,
            headers=tenant_authorization_header,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert authenticated_response.status_code != 401, (
            f"The authenticated tenant was rejected with 401 at {praxis_responses_url}; the public boundary "
            "rejects every caller, so an unauthenticated 401 would prove nothing"
        )

        response = request_session.post(url=praxis_responses_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        assert response.status_code == 401, (
            f"Expected HTTP 401 for an unauthenticated POST to {praxis_responses_url}, got {response.status_code}"
        )

    def test_malformed_bearer_token_is_rejected(
        self,
        request_session: requests.Session,
        praxis_responses_url: str,
        ogx_server: OgxServer,
    ) -> None:
        """A malformed bearer token is rejected at the public Responses API.

        Given: Praxis serves the Responses API at the public Gateway boundary.
        When: A POST is sent with an Authorization header carrying a malformed bearer token.
        Then: The response is HTTP 401.
        """
        response = request_session.post(
            url=praxis_responses_url,
            headers={"Authorization": MALFORMED_BEARER_TOKEN, "Content-Type": "application/json"},
            json={"model": "malformed-token-probe", "input": "ping"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert response.status_code == 401, (
            f"Expected HTTP 401 for a malformed bearer token at {praxis_responses_url}, got {response.status_code}"
        )
