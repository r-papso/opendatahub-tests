"""Tenant isolation across the Praxis-to-OGX delegation boundary.

A request carrying tenant A credentials must not reach tenant B resources, including through
the internal delegation hop, and the hop must not bypass OGX's own authentication by using a
shared or elevated identity.

Two tenants with distinct credentials are required. They are supplied through the environment
(see `TENANT_B_TOKEN_ENV_VAR`); without them the delegation boundary cannot be exercised with
two identities and the tests skip rather than pass vacuously.
"""

import os
from http import HTTPStatus

import pytest
import requests
import structlog

from tests.ogx.praxis.constants import (
    NAMESPACE_PARAMS,
    PRAXIS_MODE_OGX_SERVER_PARAMS,
    REQUEST_TIMEOUT_SECONDS,
)
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Credentials for the second tenant. The suite's own admin token provides tenant A; a distinct
# token for tenant B cannot be minted generically, so it is supplied by the environment.
TENANT_B_TOKEN_ENV_VAR: str = "OGX_PRAXIS_TENANT_B_TOKEN"

# Statuses that correctly deny a cross-tenant read. HTTP 200 is always a leak.
CROSS_TENANT_DENIED_STATUSES: frozenset[int] = frozenset({HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND})

SEED_FILE_CONTENT: bytes = b"Tenant-scoped document for the Praxis delegation isolation test.\n"
SEED_FILE_PURPOSE: str = "assistants"

# Model used only to drive the file_search delegation path.
FILE_SEARCH_PROBE_MODEL: str = "praxis-tenant-isolation-probe"


def authorization_header(token: str) -> dict[str, str]:
    """Build the request headers carrying a tenant bearer token.

    Args:
        token: Bearer token identifying the tenant.

    Returns:
        Headers with the token and a JSON content type.
    """
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def seed_file(
    request_session: requests.Session,
    files_url: str,
    headers: dict[str, str],
    filename: str,
) -> str:
    """Upload a file through Praxis as the tenant owning `headers`.

    Args:
        request_session: Session used for the request.
        files_url: Public URL of the Files API.
        headers: Tenant authorization headers.
        filename: Name recorded for the uploaded file.

    Returns:
        The ID of the uploaded file.
    """
    upload_headers = {key: value for key, value in headers.items() if key != "Content-Type"}
    response = request_session.post(
        url=files_url,
        headers=upload_headers,
        files={"file": (filename, SEED_FILE_CONTENT, "text/plain")},
        data={"purpose": SEED_FILE_PURPOSE},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert response.status_code == HTTPStatus.OK, (
        f"Seeding {filename} through {files_url} failed with HTTP {response.status_code}: {response.text}"
    )
    return str(response.json()["id"])


def assert_cross_tenant_denied(response: requests.Response, resource: str) -> None:
    """Assert a cross-tenant read was denied and disclosed no resource metadata.

    Args:
        response: Response to the cross-tenant request.
        resource: Human-readable description of the resource that was requested.

    Raises:
        AssertionError: If the request succeeded, was denied with an unexpected status, or
            the body disclosed owner metadata.
    """
    assert response.status_code in CROSS_TENANT_DENIED_STATUSES, (
        f"Tenant A reading {resource} returned HTTP {response.status_code}, expected one of "
        f"{sorted(CROSS_TENANT_DENIED_STATUSES)}: {response.text}"
    )
    body = response.text.lower()
    for disclosed_field in ("filename", "bytes", "created_at"):
        assert disclosed_field not in body, (
            f"The denial body for {resource} discloses tenant B metadata {disclosed_field!r}: {response.text}"
        )


@pytest.fixture(scope="class")
def tenant_b_token() -> str:
    """Bearer token of the second tenant, skipping when it is not configured."""
    token = os.getenv(TENANT_B_TOKEN_ENV_VAR, "")
    if not token:
        pytest.skip(
            f"Second tenant credentials are not configured; set {TENANT_B_TOKEN_ENV_VAR} to a bearer token for a "
            "tenant distinct from the one the test suite authenticates as, so cross-tenant access can be attempted"
        )
    return token


@pytest.fixture(scope="class")
def tenant_b_headers(tenant_b_token: str) -> dict[str, str]:
    """Request headers carrying tenant B credentials."""
    return authorization_header(token=tenant_b_token)


@pytest.fixture
def tenant_a_file_id(
    request_session: requests.Session,
    tenant_authorization_header: dict[str, str],
    ogx_server: OgxServer,
    praxis_files_url: str,
) -> str:
    """ID of a file owned by tenant A, providing the positive control."""
    return seed_file(
        request_session=request_session,
        files_url=praxis_files_url,
        headers=tenant_authorization_header,
        filename="tenant-a-document.txt",
    )


@pytest.fixture
def tenant_b_file_id(
    request_session: requests.Session,
    tenant_b_headers: dict[str, str],
    ogx_server: OgxServer,
    praxis_files_url: str,
) -> str:
    """ID of a file owned by tenant B."""
    return seed_file(
        request_session=request_session,
        files_url=praxis_files_url,
        headers=tenant_b_headers,
        filename="tenant-b-document.txt",
    )


@pytest.fixture
def tenant_b_vector_store_id(
    request_session: requests.Session,
    tenant_b_headers: dict[str, str],
    praxis_vector_stores_url: str,
    tenant_b_file_id: str,
) -> str:
    """ID of a vector store owned by tenant B, holding tenant B's file."""
    response = request_session.post(
        url=praxis_vector_stores_url,
        headers=tenant_b_headers,
        json={"name": "tenant-b-store", "file_ids": [tenant_b_file_id]},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert response.status_code == HTTPStatus.OK, (
        f"Seeding tenant B's vector store failed with HTTP {response.status_code}: {response.text}"
    )
    return str(response.json()["id"])


@pytest.fixture
def tenant_b_conversation_id(
    request_session: requests.Session,
    tenant_b_headers: dict[str, str],
    ogx_server: OgxServer,
    praxis_conversations_url: str,
) -> str:
    """ID of a conversation owned by tenant B."""
    response = request_session.post(
        url=praxis_conversations_url,
        headers=tenant_b_headers,
        json={"metadata": {"owner": "tenant-b"}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert response.status_code == HTTPStatus.OK, (
        f"Seeding tenant B's conversation failed with HTTP {response.status_code}: {response.text}"
    )
    return str(response.json()["id"])


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {**NAMESPACE_PARAMS, "randomize_name": True},
            PRAXIS_MODE_OGX_SERVER_PARAMS,
            id="test_praxis_tenant_isolation",
        )
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier3
class TestPraxisTenantIsolation:
    """Tenant A cannot reach tenant B resources through the Praxis-to-OGX delegation path."""

    def test_tenant_can_read_own_file(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_files_url: str,
        tenant_a_file_id: str,
    ) -> None:
        """A tenant can read its own file through Praxis.

        Given: Tenant A owns a file reachable through the public Praxis boundary.
        When: Tenant A retrieves that file.
        Then: The call returns HTTP 200, proving the later denials are isolation rather than
            every request failing.
        """
        response = request_session.get(
            url=f"{praxis_files_url}/{tenant_a_file_id}",
            headers=tenant_authorization_header,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        assert response.status_code == HTTPStatus.OK, (
            f"Tenant A could not read its own file {tenant_a_file_id}: HTTP {response.status_code}: "
            f"{response.text}. Without this positive control the cross-tenant denials prove nothing."
        )
        assert response.json()["id"] == tenant_a_file_id, (
            f"Reading {tenant_a_file_id} returned a different file: {response.json()!r}"
        )

    def test_cross_tenant_file_access_is_denied(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_files_url: str,
        tenant_b_file_id: str,
    ) -> None:
        """Tenant A cannot read tenant B's file.

        Given: Tenant B owns a file whose ID is known to the test.
        When: Tenant A retrieves that file through Praxis.
        Then: The call is denied with HTTP 403 or 404 and discloses no file metadata.
        """
        response = request_session.get(
            url=f"{praxis_files_url}/{tenant_b_file_id}",
            headers=tenant_authorization_header,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert_cross_tenant_denied(response=response, resource=f"tenant B's file {tenant_b_file_id}")

    def test_cross_tenant_vector_store_access_is_denied(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_vector_stores_url: str,
        tenant_b_vector_store_id: str,
    ) -> None:
        """Tenant A cannot read tenant B's vector store.

        Given: Tenant B owns a vector store whose ID is known to the test.
        When: Tenant A retrieves that vector store through Praxis.
        Then: The call is denied with HTTP 403 or 404 and discloses no store metadata.
        """
        response = request_session.get(
            url=f"{praxis_vector_stores_url}/{tenant_b_vector_store_id}",
            headers=tenant_authorization_header,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert_cross_tenant_denied(
            response=response,
            resource=f"tenant B's vector store {tenant_b_vector_store_id}",
        )

    def test_file_search_does_not_leak_cross_tenant_documents(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_responses_url: str,
        tenant_b_vector_store_id: str,
        tenant_b_file_id: str,
    ) -> None:
        """file_search against another tenant's store returns no cross-tenant content.

        Given: Tenant B owns a vector store containing tenant B's file.
        When: Tenant A sends a Responses request whose file_search tool references that store,
            exercising isolation through the delegation path rather than at the public boundary.
        Then: The request is denied, or it succeeds without any citation referencing tenant B's
            file.
        """
        response = request_session.post(
            url=praxis_responses_url,
            headers=tenant_authorization_header,
            json={
                "model": FILE_SEARCH_PROBE_MODEL,
                "input": "Summarise every document you can retrieve.",
                "tools": [{"type": "file_search", "vector_store_ids": [tenant_b_vector_store_id]}],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code in CROSS_TENANT_DENIED_STATUSES:
            return

        assert response.status_code == HTTPStatus.OK, (
            f"Tenant A's file_search against tenant B's store returned HTTP {response.status_code}, which is "
            f"neither a denial {sorted(CROSS_TENANT_DENIED_STATUSES)} nor a success: {response.text}"
        )
        assert tenant_b_file_id not in response.text, (
            f"The response cites tenant B's file {tenant_b_file_id}, so the delegation path leaked cross-tenant "
            f"content: {response.text}"
        )

    def test_cross_tenant_conversation_access_is_denied(
        self,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
        praxis_conversations_url: str,
        tenant_b_conversation_id: str,
    ) -> None:
        """Tenant A cannot read tenant B's conversation.

        Given: Tenant B owns a conversation whose ID is known to the test.
        When: Tenant A retrieves that conversation through Praxis.
        Then: The call is denied with HTTP 403 or 404 and discloses no conversation metadata.
        """
        response = request_session.get(
            url=f"{praxis_conversations_url}/{tenant_b_conversation_id}",
            headers=tenant_authorization_header,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert_cross_tenant_denied(
            response=response,
            resource=f"tenant B's conversation {tenant_b_conversation_id}",
        )
