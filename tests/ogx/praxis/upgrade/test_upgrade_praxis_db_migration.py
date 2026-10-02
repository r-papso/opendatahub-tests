"""OGX -> Praxis database migration coverage for the upgrade pipeline.

The pre-upgrade test seeds the OGX responses and conversations tables through the
OGX API, and brings up the Praxis (target) PostgreSQL instance together with the
Secret holding its connection string. The upgrade itself -- enabling
`spec.praxisMode` and pointing `migrationJob.targetConnectionString` at that
Secret -- is performed by the Jenkins job that runs these tests.
The post-upgrade test waits for the operator-created migration Job and then
compares both tables row by row, reading the two PostgreSQL databases directly so
that neither OGX nor Praxis needs to be serving traffic.

Both tests emit a JSON report with before/after row counts and the active
embedding model, for retention as a CI artifact.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ogx_client import OgxClient

from tests.ogx.constants import ModelInfo
from tests.ogx.praxis.constants import (
    OGX_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_DATABASE,
    SEED_CONVERSATIONS_COUNT,
    SEED_RESPONSES_COUNT,
    SOURCE_CONVERSATIONS_TABLE,
    SOURCE_RESPONSES_TABLE,
    TARGET_CONVERSATIONS_TABLE,
    TARGET_RESPONSES_TABLE,
)
from tests.ogx.praxis.utils import (
    compare_rows,
    format_parity_failure,
    query_database,
    seed_conversations,
    seed_responses,
    wait_for_migration_job_completion,
)
from utilities.resources.ogx_server import OgxServer

# `metadata` is a JSON column in OGX and a JSON-serialized text column in Praxis;
# routing both through `jsonb` canonicalizes key order and whitespace so the two
# representations are comparable.
_JSON_METADATA_COLUMN = "COALESCE(metadata::text::jsonb, '{}'::jsonb)::text AS metadata"


def _responses_query(table_name: str) -> str:
    """Return the responses SELECT for `table_name`."""
    return f"SELECT id, created_at, model FROM {table_name}"


def _conversations_query(table_name: str, id_column: str) -> str:
    """Return the conversations SELECT for `table_name`, whose id column differs between OGX and Praxis."""
    return f"SELECT {id_column}, created_at, {_JSON_METADATA_COLUMN} FROM {table_name}"


SOURCE_RESPONSES_QUERY = _responses_query(table_name=SOURCE_RESPONSES_TABLE)
SOURCE_CONVERSATIONS_QUERY = _conversations_query(table_name=SOURCE_CONVERSATIONS_TABLE, id_column="id")
TARGET_RESPONSES_QUERY = _responses_query(table_name=TARGET_RESPONSES_TABLE)
TARGET_CONVERSATIONS_QUERY = _conversations_query(table_name=TARGET_CONVERSATIONS_TABLE, id_column="conversation_id")

# Neither `spec.praxisMode` nor `spec.storage.sql` is set here: both belong to
# the upgrade step. The operator creates the migration Job as soon as praxisMode
# is present, which pre-upgrade would mean migrating the tables before this test
# has seeded them, and setting `spec.storage` switches the server onto the
# operator's generated config, replacing the storage section the distribution
# derives from the POSTGRES_* environment (including moving the KV store to
# sqlite, which the CRD cannot express as postgres).
OGX_SERVER_PARAMS: dict[str, Any] = {"vector_io_provider": "faiss", "files_provider": "local"}
NAMESPACE_PARAMS = {"name": "test-ogx-praxis-db-migration"}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
class TestPreUpgradePraxisDbMigration:
    @pytest.mark.pre_upgrade
    def test_seed_ogx_responses_and_conversations_pre_upgrade(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        ogx_server: OgxServer,
        ogx_postgres_pod: Pod,
        praxis_connection_string_secret: Secret,
    ) -> None:
        """Populate the OGX responses and conversations tables before the upgrade.

        Given: A running OGX distribution backed by PostgreSQL, before the Praxis upgrade.
        When: Conversations and stored responses are created through the OGX API.
        Then: The rows are readable in the OGX conversations and responses tables,
            and the before-upgrade counts are reported. The Praxis database and its
            connection-string Secret are left in place for the upgrade step.
        """
        conversation_ids = seed_conversations(ogx_client=ogx_client, count=SEED_CONVERSATIONS_COUNT)
        response_ids = seed_responses(
            ogx_client=ogx_client,
            model_id=ogx_models.model_id,
            count=SEED_RESPONSES_COUNT,
            conversation_id=conversation_ids[0],
        )

        source_responses = query_database(
            postgres_pod=ogx_postgres_pod,
            database=OGX_POSTGRES_DATABASE,
            select_statement=SOURCE_RESPONSES_QUERY,
        )
        source_conversations = query_database(
            postgres_pod=ogx_postgres_pod,
            database=OGX_POSTGRES_DATABASE,
            select_statement=SOURCE_CONVERSATIONS_QUERY,
        )

        stored_response_ids = {str(row["id"]) for row in source_responses}
        assert not set(response_ids) - stored_response_ids, (
            f"Responses created through the API are missing from {SOURCE_RESPONSES_TABLE}: "
            f"{sorted(set(response_ids) - stored_response_ids)}"
        )
        stored_conversation_ids = {str(row["id"]) for row in source_conversations}
        assert not set(conversation_ids) - stored_conversation_ids, (
            f"Conversations created through the API are missing from {SOURCE_CONVERSATIONS_TABLE}: "
            f"{sorted(set(conversation_ids) - stored_conversation_ids)}"
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
class TestPostUpgradePraxisDbMigration:
    @pytest.mark.post_upgrade
    def test_responses_and_conversations_migrated_to_praxis_post_upgrade(
        self,
        unprivileged_client: DynamicClient,
        ogx_server: OgxServer,
        ogx_postgres_pod: Pod,
        configured_migration_target: dict[str, str],
        praxis_connection_string_secret: Secret,
        praxis_postgres_pod: Pod,
    ) -> None:
        """Verify the migration Job copied the OGX responses and conversations into Praxis.

        Given: An upgraded OGX distribution with `praxisMode.migrationJob` configured,
            whose source database still holds the rows seeded before the upgrade.
        When: The operator-created migration Job has completed and both databases
            are read directly.
        Then: Every source row is present in the Praxis table with matching field
            values. Rows present only in Praxis are tolerated for conversations,
            where the migration synthesizes entries for message-only conversations.
        """
        wait_for_migration_job_completion(client=unprivileged_client, ogx_server=ogx_server)

        responses_parity = compare_rows(
            table=SOURCE_RESPONSES_TABLE,
            source_rows=query_database(
                postgres_pod=ogx_postgres_pod,
                database=OGX_POSTGRES_DATABASE,
                select_statement=SOURCE_RESPONSES_QUERY,
            ),
            target_rows=query_database(
                postgres_pod=praxis_postgres_pod,
                database=PRAXIS_POSTGRES_DATABASE,
                select_statement=TARGET_RESPONSES_QUERY,
            ),
            source_key="id",
            target_key="id",
            compared_fields=("created_at", "model"),
        )
        conversations_parity = compare_rows(
            table=SOURCE_CONVERSATIONS_TABLE,
            source_rows=query_database(
                postgres_pod=ogx_postgres_pod,
                database=OGX_POSTGRES_DATABASE,
                select_statement=SOURCE_CONVERSATIONS_QUERY,
            ),
            target_rows=query_database(
                postgres_pod=praxis_postgres_pod,
                database=PRAXIS_POSTGRES_DATABASE,
                select_statement=TARGET_CONVERSATIONS_QUERY,
            ),
            source_key="id",
            target_key="conversation_id",
            compared_fields=("created_at", "metadata"),
        )

        assert responses_parity.source_count, (
            f"No rows found in the source {SOURCE_RESPONSES_TABLE} table; the pre-upgrade test did not seed "
            "data, so migration parity cannot be asserted"
        )
        assert conversations_parity.source_count, (
            f"No rows found in the source {SOURCE_CONVERSATIONS_TABLE} table; the pre-upgrade test did not seed "
            "data, so migration parity cannot be asserted"
        )
        failures = [
            format_parity_failure(parity=responses_parity, allow_extra_in_target=False),
            format_parity_failure(parity=conversations_parity, allow_extra_in_target=True),
        ]
        assert not any(failures), "\n".join(failure for failure in failures if failure)
