"""Write-path ownership of resources created through Praxis after the upgrade.

In the upgraded topology Praxis fronts OGX, and a resource created afterwards
must land in exactly one of the two state stores: in both it is a dual write,
in neither it was never persisted at all.

The test creates a file, a vector store, a conversation and a response through
the public API, then reads both PostgreSQL databases directly and asserts that
exactly one of them holds each new id. Which of the two owns a given resource
type is deliberately not asserted: the strategy still lists that as TBD, so the
test pins the invariant rather than the current answer. Both instances are the
ones the Praxis database-migration suite brings up, so this module reuses that
suite's namespace and OGXServer parameters; it has no pre-upgrade phase of its
own.

Not covered: the audit-log half of the test case, which asks that each id appear
in write operations against exactly one backend. This repository has no
audit-log collection harness, and an assertion over data that is never collected
would pass no matter how the write path behaved, so the direct state-store check
is implemented instead.
"""

from typing import Any

import pytest
import structlog
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import ModelInfo
from tests.ogx.praxis.upgrade.constants import (
    OGX_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_DATABASE,
    SEED_MARKER,
    SOURCE_FILES_TABLE,
    SOURCE_RESPONSES_TABLE,
    SOURCE_VECTOR_STORES_TABLE,
    TARGET_FILES_TABLE,
    TARGET_RESPONSES_TABLE,
    TARGET_VECTOR_STORES_TABLE,
)
from tests.ogx.praxis.upgrade.utils import (
    ids_present_in_table,
    seed_conversations,
    seed_files,
    seed_responses,
)
from tests.ogx.utils import vector_store_create_and_poll

LOGGER = structlog.get_logger(name=__name__)

# Mirrors the Praxis database-migration suite: that suite's pre-upgrade phase
# deploys both the OGX and the Praxis PostgreSQL instances this test reads, and
# the fixtures resolve them inside the namespace they were created in.
NAMESPACE_PARAMS = {"name": "test-ogx-praxis-db-migration"}
OGX_SERVER_PARAMS: dict[str, Any] = {"vector_io_provider": "pgvector", "files_provider": "local"}

# Must match `vector_io_provider` above: the vector store is created explicitly
# rather than through the shared `vector_store` fixture, because that fixture
# reuses the pre-upgrade store, while this test needs a store written after the
# upgrade.
VECTOR_IO_PROVIDER: str = "pgvector"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
class TestPraxisWritePathOwnership:
    """Verify post-upgrade writes reach a single authoritative state store."""

    @pytest.mark.post_upgrade
    def test_resources_written_to_single_backend(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        ogx_postgres_pod: Pod,
        praxis_postgres_pod: Pod,
    ) -> None:
        """Verify resources created through Praxis are stored by exactly one backend.

        Given: A cluster upgraded to 3.6, with Praxis fronting OGX and both the OGX and
            the Praxis PostgreSQL databases reachable.
        When: A file, a vector store with that file attached, a conversation and a
            response are created through the public API.
        Then: Every create call returns a non-empty id, the file-to-vector-store
            attachment reaches 'completed', and each id is found in exactly one of the
            two databases -- never in both, never in neither.
        """
        file_id = seed_files(ogx_client=ogx_client, count=1)[0]
        assert file_id, "Creating a file returned an empty id"

        vector_store = ogx_client.vector_stores.create(
            name=f"{SEED_MARKER}-write-path",
            extra_body={
                "embedding_model": ogx_models.embedding_model.id,
                "embedding_dimension": ogx_models.embedding_dimension,
                "provider_id": VECTOR_IO_PROVIDER,
            },
        )
        assert vector_store.id, "Creating a vector store returned an empty id"

        vector_store_file = vector_store_create_and_poll(
            ogx_client=ogx_client,
            vector_store_id=vector_store.id,
            file_id=file_id,
        )
        assert vector_store_file.status == "completed", (
            f"File {file_id} did not finish attaching to vector store {vector_store.id}: "
            f"status={vector_store_file.status!r}, last_error={vector_store_file.last_error!r}"
        )

        conversation_id = seed_conversations(ogx_client=ogx_client, count=1)[0]
        assert conversation_id, "Creating a conversation returned an empty id"

        response_id = seed_responses(
            ogx_client=ogx_client,
            model_id=ogx_models.model_id,
            count=1,
            conversation_id=conversation_id,
        )[0]
        assert response_id, "Creating a response returned an empty id"

        # Resource, its id, the OGX and Praxis tables to look that id up in, and
        # the Praxis identity column. The OGX side is always keyed on `id`.
        ownership_checks = (
            ("File", file_id, SOURCE_FILES_TABLE, TARGET_FILES_TABLE, "id"),
            ("Vector store", vector_store.id, SOURCE_VECTOR_STORES_TABLE, TARGET_VECTOR_STORES_TABLE, "id"),
            ("Response", response_id, SOURCE_RESPONSES_TABLE, TARGET_RESPONSES_TABLE, "id"),
        )

        for resource, resource_id, ogx_table, praxis_table, praxis_id_column in ownership_checks:
            holders = {
                f"OGX '{OGX_POSTGRES_DATABASE}.{ogx_table}'": ids_present_in_table(
                    postgres_pod=ogx_postgres_pod,
                    database=OGX_POSTGRES_DATABASE,
                    table=ogx_table,
                    ids=[resource_id],
                ),
                f"Praxis '{PRAXIS_POSTGRES_DATABASE}.{praxis_table}'": ids_present_in_table(
                    postgres_pod=praxis_postgres_pod,
                    database=PRAXIS_POSTGRES_DATABASE,
                    table=praxis_table,
                    ids=[resource_id],
                    id_column=praxis_id_column,
                ),
            }
            owners = sorted(backend for backend, found in holders.items() if found)

            assert len(owners) == 1, (
                f"{resource} '{resource_id}' must be stored by exactly one of {sorted(holders)}, but "
                f"{len(owners)} hold it: {owners or 'neither'}"
            )
            LOGGER.info(f"{resource} '{resource_id}' is stored by {owners[0]} alone")
