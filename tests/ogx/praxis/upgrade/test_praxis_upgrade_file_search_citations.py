"""TC-E2E-005: file_search citations after the OGX -> Praxis upgrade.

The pre-upgrade class records the ids of the files the `vector_store` fixture
ingested into the vector store. The post-upgrade test asks a question answered
by that corpus through `POST /v1/responses` with the file_search tool, and
checks that the citations still reference those pre-upgrade file ids.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ogx_client import OgxClient
from ogx_client.types import ResponseObject
from ogx_client.types.vector_store import VectorStore

from tests.ogx.constants import ModelInfo
from tests.ogx.datasets import IBM_2025_Q4_EARNINGS
from tests.ogx.praxis.constants import NAMESPACE_PARAMS, OGX_SERVER_PARAMS
from tests.ogx.praxis.upgrade.constants import (
    CITATION_INSTRUCTIONS,
    CITATION_MAX_OUTPUT_TOKENS,
    CITATION_QUESTION,
    FILE_SEARCH_CITATIONS_CONFIG_MAP_KEY,
)
from tests.ogx.praxis.upgrade.utils import (
    FileSearchCitationBaseline,
    load_baseline_section,
    save_baseline_section,
)


def _cited_file_ids(response: ResponseObject) -> set[str]:
    """Return the file ids referenced by the `file_citation` annotations of a response.

    Args:
        response: The `POST /v1/responses` response.

    Returns:
        The distinct cited file ids; empty when the response carries no
        file_citation annotation.
    """
    return {
        annotation.file_id
        for output_item in response.output
        if output_item.type == "message"
        for content_item in output_item.content
        for annotation in content_item.annotations
        if annotation.type == "file_citation"
    }


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            NAMESPACE_PARAMS,
            OGX_SERVER_PARAMS,
            {"vector_io_provider": "pgvector", "dataset": IBM_2025_Q4_EARNINGS},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.rag
class TestPreUpgradeFileSearchCitations:
    @pytest.mark.pre_upgrade
    def test_record_file_search_citation_baseline(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_client: OgxClient,
        vector_store: VectorStore,
    ) -> None:
        """Record the ids of the files whose citations are verified after the upgrade.

        Given: An OGX distribution before the upgrade, with the dataset ingested
            into a vector store.
        When: The attached file ids are read back and persisted to the baseline
            ConfigMap.
        Then: The vector store holds at least one file and the persisted
            baseline reads back unchanged.
        """
        file_ids = [
            vector_store_file.id
            for vector_store_file in ogx_client.vector_stores.files.list(vector_store_id=vector_store.id).data
        ]
        assert file_ids, f"Vector store {vector_store.id} has no attached files to cite after the upgrade"

        citations: FileSearchCitationBaseline = {"vector_store_id": vector_store.id, "file_ids": file_ids}
        save_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=FILE_SEARCH_CITATIONS_CONFIG_MAP_KEY,
            payload=citations,
        )
        persisted: FileSearchCitationBaseline = load_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=FILE_SEARCH_CITATIONS_CONFIG_MAP_KEY,
        )
        assert persisted == citations, f"Persisted citation baseline {persisted} differs from the recorded {citations}"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            NAMESPACE_PARAMS,
            OGX_SERVER_PARAMS,
            {"vector_io_provider": "milvus-remote"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.rag
class TestPostUpgradeFileSearchCitations:
    @pytest.mark.post_upgrade
    def test_file_search_citations_reference_pre_upgrade_files(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        vector_store: VectorStore,
    ) -> None:
        """Verify file_search citations still reference the pre-upgrade files.

        Given: A cluster upgraded with Praxis fronting OGX, and a vector store
            whose files were ingested and recorded before the upgrade.
        When: A question answered by those files is asked through
            `POST /v1/responses` with the file_search tool.
        Then: The response carries file_citation annotations, and every cited
            file id is one recorded before the upgrade.
        """
        citations: FileSearchCitationBaseline = load_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=FILE_SEARCH_CITATIONS_CONFIG_MAP_KEY,
        )
        assert citations["vector_store_id"] == vector_store.id, (
            f"Vector store reused after the upgrade is {vector_store.id}, but the baseline was recorded for "
            f"{citations['vector_store_id']}"
        )

        response = ogx_client.responses.create(
            input=CITATION_QUESTION,
            model=ogx_models.model_id,
            instructions=CITATION_INSTRUCTIONS,
            stream=False,
            store=False,
            max_output_tokens=CITATION_MAX_OUTPUT_TOKENS,
            tool_choice="required",
            include=["file_search_call.results"],
            tools=[{"type": "file_search", "vector_store_ids": [citations["vector_store_id"]]}],
        )

        cited_file_ids = _cited_file_ids(response=response)
        assert cited_file_ids, (
            "Expected at least one file_citation annotation after the upgrade; response output types: "
            f"{[output_item.type for output_item in response.output]}"
        )

        pre_upgrade_file_ids = set(citations["file_ids"])
        assert cited_file_ids <= pre_upgrade_file_ids, (
            "Citations reference file ids that did not exist before the upgrade: "
            f"{sorted(cited_file_ids - pre_upgrade_file_ids)}"
        )
