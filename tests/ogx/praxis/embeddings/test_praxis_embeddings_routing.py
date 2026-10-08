"""Routing of the `/v1/embeddings` API after the OGX -> Praxis migration.

On a greenfield RHOAI 3.6 cluster the Embeddings API is exposed through the
Platform Gateway and must be routed by Praxis to the configured embedding
backend. The serving workload is resolved from the cluster by following the
matching HTTPRoute to its backend Service and from there to its pods, so a route
still pointing at OGX cannot pass unnoticed.

The expected vector dimension is read from the backend's model metadata through
`GET /v1/models` before any embedding is requested, so the length assertion
compares against a value established independently of the embeddings response.
"""

from typing import Any, Final

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from tests.ogx.constants import OGX_CORE_EMBEDDING_MODEL, OGX_CORE_POD_FILTER
from tests.ogx.praxis.constants import (
    EMBEDDINGS_API_PATH,
    MODELS_API_PATH,
    REQUEST_TIMEOUT_SECONDS,
)
from tests.ogx.praxis.utils import gateway_base_url, http_routes_matching_path, serving_pods_for_path
from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)

# Key under which the embedding backends report the output vector length in the
# model metadata returned by `GET /v1/models`.
EMBEDDING_DIMENSION_KEY: Final[str] = "embedding_dimension"

# Metadata fields of a model entry, in the order they are looked up. OGX reports
# `custom_metadata`; the OpenAI-compatible shape uses `metadata`.
MODEL_METADATA_KEYS: Final[tuple[str, ...]] = ("custom_metadata", "metadata")

EMBEDDINGS_INPUT: Final[str] = "quarterly migration readiness review"
EMBEDDINGS_BATCH_INPUT: Final[list[str]] = [
    "quarterly migration readiness review",
    "greenfield deployment acceptance criteria",
]


def _model_metadata(model_entry: dict[str, Any]) -> dict[str, Any]:
    """Return the metadata mapping of a `GET /v1/models` entry.

    Args:
        model_entry: One element of the models listing.

    Returns:
        The first metadata mapping present on the entry, or an empty mapping.
    """
    for metadata_key in MODEL_METADATA_KEYS:
        metadata = model_entry.get(metadata_key)
        if isinstance(metadata, dict):
            return metadata
    return {}


@pytest.fixture
def embeddings_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """The HTTPRoute that declares the `/v1/embeddings` path."""
    matching_routes = http_routes_matching_path(client=admin_client, path=EMBEDDINGS_API_PATH)
    if len(matching_routes) != 1:
        pytest.fail(
            f"Expected exactly one HTTPRoute declaring {EMBEDDINGS_API_PATH}, found "
            f"{[f'{route.namespace}/{route.name}' for route in matching_routes]}"
        )
    return matching_routes[0]


@pytest.fixture
def embeddings_model_dimension(
    embeddings_http_route: HTTPRoute,
    request_session: requests.Session,
    tenant_authorization_header: dict[str, str],
) -> int:
    """Output vector length the backend reports for the configured embedding model."""
    models_url = f"{gateway_base_url(http_route=embeddings_http_route)}{MODELS_API_PATH}"
    response = request_session.get(
        url=models_url,
        headers=tenant_authorization_header,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        pytest.skip(
            f"GET {MODELS_API_PATH} returned HTTP {response.status_code}, so the expected embedding "
            "dimension cannot be established independently of the embeddings response"
        )

    model_entry = next(
        (entry for entry in response.json().get("data") or [] if entry.get("id") == OGX_CORE_EMBEDDING_MODEL),
        None,
    )
    if model_entry is None:
        pytest.fail(
            f"Embedding model '{OGX_CORE_EMBEDDING_MODEL}' is not registered at {MODELS_API_PATH}; "
            "the embedding backend is not reachable through Praxis"
        )

    dimension = _model_metadata(model_entry=model_entry).get(EMBEDDING_DIMENSION_KEY)
    if dimension is None:
        pytest.skip(
            f"Model metadata of '{OGX_CORE_EMBEDDING_MODEL}' does not publish '{EMBEDDING_DIMENSION_KEY}', "
            "so the expected vector length cannot be established independently of the embeddings response"
        )

    LOGGER.info(f"Expected embedding dimension of {OGX_CORE_EMBEDDING_MODEL}: {dimension}")
    return int(dimension)


@pytest.mark.ogx
@pytest.mark.skip_must_gather
class TestPraxisEmbeddingsRouting:
    """Verify Praxis routes `/v1/embeddings` to the configured embedding backend."""

    @pytest.mark.tier1
    def test_embeddings_routed_by_praxis_to_embedding_backend(
        self,
        admin_client: DynamicClient,
        embeddings_http_route: HTTPRoute,
        embeddings_model_dimension: int,
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
    ) -> None:
        """Verify the Embeddings API is routed by Praxis to the embedding backend.

        Given a greenfield cluster exposing the Embeddings API through the Platform Gateway,
        When authenticated POST /v1/embeddings requests with a single input and with a two-element
        input array are sent through the external Gateway hostname,
        Then no OGX pod backs the route, both requests succeed, the returned vector has the length
        the backend publishes for the configured model, the echoed model matches the requested one,
        the batch response carries exactly two indexed objects and prompt tokens are accounted for.
        """
        serving_pods = serving_pods_for_path(
            client=admin_client, http_route=embeddings_http_route, path=EMBEDDINGS_API_PATH
        )
        ogx_pod_keys = {
            (pod.namespace, pod.name) for pod in Pod.get(client=admin_client, label_selector=OGX_CORE_POD_FILTER)
        }
        serving_ogx_pods = sorted(
            f"{pod.namespace}/{pod.name}" for pod in serving_pods if (pod.namespace, pod.name) in ogx_pod_keys
        )
        assert not serving_ogx_pods, (
            f"{EMBEDDINGS_API_PATH} is backed by OGX pods {serving_ogx_pods}; it must be routed by Praxis"
        )

        embeddings_url = f"{gateway_base_url(http_route=embeddings_http_route)}{EMBEDDINGS_API_PATH}"
        LOGGER.info(f"POST {embeddings_url} with model {OGX_CORE_EMBEDDING_MODEL}")
        single_response = request_session.post(
            url=embeddings_url,
            headers=tenant_authorization_header,
            json={"model": OGX_CORE_EMBEDDING_MODEL, "input": EMBEDDINGS_INPUT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert single_response.status_code == 200, (
            f"POST {EMBEDDINGS_API_PATH} returned HTTP {single_response.status_code}: {single_response.text[:200]}"
        )

        single_body = single_response.json()
        embedding = single_body["data"][0]["embedding"]
        assert len(embedding) == embeddings_model_dimension, (
            f"Embedding has {len(embedding)} elements, but the backend reports "
            f"{embeddings_model_dimension} for model {OGX_CORE_EMBEDDING_MODEL}"
        )
        assert single_body.get("model") == OGX_CORE_EMBEDDING_MODEL, (
            f"Response was produced by model {single_body.get('model')!r}, "
            f"not the requested {OGX_CORE_EMBEDDING_MODEL!r}"
        )
        assert single_body["usage"]["prompt_tokens"] > 0, (
            f"Embeddings response reports no prompt tokens: {single_body.get('usage')}"
        )

        batch_response = request_session.post(
            url=embeddings_url,
            headers=tenant_authorization_header,
            json={"model": OGX_CORE_EMBEDDING_MODEL, "input": EMBEDDINGS_BATCH_INPUT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert batch_response.status_code == 200, (
            f"Batch POST {EMBEDDINGS_API_PATH} returned HTTP {batch_response.status_code}: {batch_response.text[:200]}"
        )

        batch_data = batch_response.json()["data"]
        assert [entry["index"] for entry in batch_data] == [0, 1], (
            f"Batch of {len(EMBEDDINGS_BATCH_INPUT)} inputs returned indexes "
            f"{[entry.get('index') for entry in batch_data]}, expected exactly [0, 1]"
        )
