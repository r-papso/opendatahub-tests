"""TC-E2E-005: file_search citations after an OGX -> Praxis migration.

The module ingests a dataset into OGX, records the vector store and attached
file ids, and then enables Praxis mode to run the migration. The post-migration
request goes through the external Gateway hostname Praxis serves and verifies
that file_search citations still reference the files ingested through OGX.
"""

from collections.abc import Callable

import pytest
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
)
from tests.ogx.praxis.upgrade.utils import FileSearchCitationBaseline
from utilities.resources.ogx_server import OgxServer


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


@pytest.fixture(scope="class")
def file_search_citations_baseline(
    ogx_client: OgxClient,
    vector_store: VectorStore,
    praxis_migration_runner: Callable[[], OgxServer],
) -> FileSearchCitationBaseline:
    """Record OGX file ids and run the Praxis migration once."""
    file_ids = [
        vector_store_file.id
        for vector_store_file in ogx_client.vector_stores.files.list(vector_store_id=vector_store.id).data
    ]
    assert file_ids, f"Vector store {vector_store.id} has no attached files to cite after migration"

    baseline: FileSearchCitationBaseline = {"vector_store_id": vector_store.id, "file_ids": file_ids}
    praxis_migration_runner()
    return baseline


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
class TestFileSearchCitationsMigration:
    def test_file_search_citations_reference_migrated_files(
        self,
        file_search_citations_baseline: FileSearchCitationBaseline,
        praxis_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """Verify file_search citations still reference the migrated OGX files.

        Given: A vector store whose files were ingested through OGX and migrated
            to Praxis.
        When: A question answered by those files is asked through
            `POST /v1/responses` with the file_search tool.
        Then: The migrated vector store is served through Praxis, the
            response carries file_citation annotations, and every cited file id
            is one recorded before migration.
        """
        baseline_vector_store_id = file_search_citations_baseline["vector_store_id"]
        vector_store = praxis_client.vector_stores.retrieve(vector_store_id=baseline_vector_store_id)
        assert vector_store.id == baseline_vector_store_id, (
            f"Retrieving vector store {baseline_vector_store_id} through Praxis returned {vector_store.id}"
        )

        response = praxis_client.responses.create(
            input=CITATION_QUESTION,
            model=ogx_models.model_id,
            instructions=CITATION_INSTRUCTIONS,
            stream=False,
            store=False,
            max_output_tokens=CITATION_MAX_OUTPUT_TOKENS,
            tool_choice="required",
            include=["file_search_call.results"],
            tools=[{"type": "file_search", "vector_store_ids": [baseline_vector_store_id]}],
        )

        cited_file_ids = _cited_file_ids(response=response)
        assert cited_file_ids, (
            "Expected at least one file_citation annotation after migration; response output types: "
            f"{[output_item.type for output_item in response.output]}"
        )

        pre_migration_file_ids = set(file_search_citations_baseline["file_ids"])
        assert cited_file_ids <= pre_migration_file_ids, (
            "Citations reference file ids that did not exist before migration: "
            f"{sorted(cited_file_ids - pre_migration_file_ids)}"
        )
