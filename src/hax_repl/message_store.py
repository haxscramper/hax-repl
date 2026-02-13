from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from hax_repl.models import (AnyMessage, ContextHashID, PromptMessage,
                             ResponseMessage)


class MessageStore:

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    context_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """)
            conn.commit()

    def upsert(self, message: AnyMessage) -> None:
        payload_json = message.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (context_id, kind, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(context_id) DO UPDATE SET
                    kind = excluded.kind,
                    payload_json = excluded.payload_json
                """,
                (message.context_id.md5, message.kind, payload_json),
            )
            conn.commit()

    def get(self, context_id: ContextHashID) -> AnyMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT kind, payload_json FROM messages WHERE context_id = ?",
                (context_id.md5, ),
            ).fetchone()
        if row is None:
            return None
        if row["kind"] == "prompt":
            return PromptMessage.model_validate_json(row["payload_json"])
        if row["kind"] == "response":
            return ResponseMessage.model_validate_json(row["payload_json"])
        raise RuntimeError(f"Unknown message kind in sqlite: {row['kind']}")

    def get_many(self,
                 context_ids: Sequence[ContextHashID]) -> list[AnyMessage]:
        messages: list[AnyMessage] = []
        for context_id in context_ids:
            message = self.get(context_id)
            if message is not None:
                messages.append(message)
        return messages

    def delete(self, context_id: ContextHashID) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE context_id = ?",
                         (context_id.md5, ))
            conn.commit()
