from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from langchain_core.runnables import RunnableConfig


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "checkpoints.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class SQLiteSaver(BaseCheckpointSaver):
    """Lightweight SQLite-backed checkpoint saver.

    Note: This implements a subset of the `InMemorySaver` behavior sufficient for
    persisting/retrieving checkpoints for typical agent usage. It stores serialized
    blobs using the serializer from `BaseCheckpointSaver`.
    """

    def __init__(self, db_path: str | Path | None = None, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT,
                checkpoint_ns TEXT,
                checkpoint_id TEXT,
                type TEXT,
                blob BLOB,
                metadata_type TEXT,
                metadata_blob BLOB,
                parent_checkpoint_id TEXT,
                PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                thread_id TEXT,
                checkpoint_ns TEXT,
                channel TEXT,
                version TEXT,
                type TEXT,
                blob BLOB,
                PRIMARY KEY(thread_id, checkpoint_ns, channel, version)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT,
                checkpoint_ns TEXT,
                checkpoint_id TEXT,
                task_id TEXT,
                write_idx INTEGER,
                channel TEXT,
                type TEXT,
                blob BLOB,
                task_path TEXT,
                PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- helpers ----
    def _serialize(self, obj: Any) -> tuple[str, bytes]:
        return self.serde.dumps_typed(obj)

    def _deserialize(self, type_blob: tuple[str, bytes]) -> Any:
        return self.serde.loads_typed(type_blob)

    # ---- required interface ----
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        cur = self._conn.cursor()
        if checkpoint_id:
            cur.execute(
                "SELECT type, blob, metadata_type, metadata_blob, parent_checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                (thread_id, checkpoint_ns, checkpoint_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            checkpoint_t, metadata_t, parent_id = (row[0:2], row[2:4], row[4])
        else:
            cur.execute(
                "SELECT checkpoint_id, type, blob, metadata_type, metadata_blob, parent_checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id, checkpoint_ns),
            )
            row = cur.fetchone()
            if not row:
                return None
            checkpoint_id = row[0]
            checkpoint_t = (row[1], row[2])
            metadata_t = (row[3], row[4])
            parent_id = row[5]

        checkpoint_obj = self._deserialize(checkpoint_t)
        metadata_obj = self._deserialize(metadata_t)

        # load any blobs for channel values
        channel_values = {}
        cur.execute(
            "SELECT channel, type, blob FROM blobs WHERE thread_id=? AND checkpoint_ns=?",
            (thread_id, checkpoint_ns),
        )
        for channel, t, b in cur.fetchall():
            channel_values[channel] = self._deserialize((t, b))

        # load pending writes (simplified)
        cur.execute(
            "SELECT task_id, write_idx, channel, type, blob FROM writes WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? ORDER BY task_id, write_idx",
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        pending_writes = []
        for task_id, write_idx, channel, t, b in cur.fetchall():
            pending_writes.append((task_id, channel, self._deserialize((t, b))))

        # attach channel_values into checkpoint
        checkpoint_obj["channel_values"] = channel_values

        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}},
            checkpoint=checkpoint_obj,
            metadata=metadata_obj,
            parent_config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}} if parent_id else None,
            pending_writes=pending_writes,
        )

    def list(self, config: RunnableConfig | None, *, filter: dict | None = None, before: RunnableConfig | None = None, limit: int | None = None) -> Iterator[CheckpointTuple]:
        # Returns checkpoints for the thread_id in config or all threads if config is None
        cur = self._conn.cursor()
        thread_ids = [config["configurable"]["thread_id"]] if config else []
        checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        for thread_id in thread_ids:
            args = [thread_id]
            q = "SELECT checkpoint_id, type, blob, metadata_type, metadata_blob, parent_checkpoint_id FROM checkpoints WHERE thread_id=?"
            if checkpoint_ns is not None:
                q += " AND checkpoint_ns=?"
                args.append(checkpoint_ns)
            q += " ORDER BY checkpoint_id DESC"
            if limit:
                q += " LIMIT ?"
                args.append(limit)
            cur.execute(q, args)
            for row in cur.fetchall():
                checkpoint_id = row[0]
                checkpoint_t = (row[1], row[2])
                metadata_t = (row[3], row[4])
                parent_id = row[5]
                checkpoint_obj = self._deserialize(checkpoint_t)
                metadata_obj = self._deserialize(metadata_t)
                yield CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}},
                    checkpoint=checkpoint_obj,
                    metadata=metadata_obj,
                    parent_config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}} if parent_id else None,
                    pending_writes=[],
                )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        # serialize
        checkpoint_t = self._serialize({k: v for k, v in checkpoint.items() if k != "channel_values"})
        metadata_t = self._serialize(metadata)
        parent = config["configurable"].get("checkpoint_id")

        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id, type, blob, metadata_type, metadata_blob, parent_checkpoint_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, checkpoint_ns, checkpoint["id"], checkpoint_t[0], checkpoint_t[1], metadata_t[0], metadata_t[1], parent),
        )

        # store channel blobs
        channel_values = checkpoint.get("channel_values", {})
        for k, v in new_versions.items():
            if k in channel_values:
                t, b = self._serialize(channel_values[k])
            else:
                t, b = ("empty", b"")
            cur.execute(
                "INSERT OR REPLACE INTO blobs(thread_id, checkpoint_ns, channel, version, type, blob) VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, k, str(v), t, b),
            )

        self._conn.commit()
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint["id"]}}

    def put_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        cur = self._conn.cursor()
        for idx, (channel, value) in enumerate(writes):
            t, b = self._serialize(value)
            cur.execute(
                "INSERT OR REPLACE INTO writes(thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx, channel, type, blob, task_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, t, b, task_path),
            )
        self._conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
        cur.execute("DELETE FROM blobs WHERE thread_id=?", (thread_id,))
        cur.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
        self._conn.commit()

    # async versions simply call sync methods
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(self, config: RunnableConfig | None, *, filter: dict | None = None, before: RunnableConfig | None = None, limit: int | None = None):
        for t in self.list(config, filter=filter, before=before, limit=limit):
            yield t

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = ""):
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)
