from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from hax_repl.models import ContextHashID, SessionFile, SessionName, SessionTurn


class SessionStore:

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def resolve_session_name(self, explicit_name: str | None) -> SessionName:
        if explicit_name:
            return SessionName(value=explicit_name)
        return SessionName(value=datetime.now(timezone.utc).isoformat())

    def _path_for(self, session_name: SessionName) -> Path:
        return self._root_dir / f"{session_name.value}.yaml"

    def load_or_create(self, session_name: SessionName) -> SessionFile:
        path = self._path_for(session_name)
        if not path.exists():
            new_session = SessionFile(
                session=session_name,
                created_at_iso=datetime.now(timezone.utc).isoformat(),
                turns=[],
                message_ids=[],
            )
            self.save(new_session)
            return new_session

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return SessionFile.model_validate(raw)

    def save(self, session: SessionFile) -> None:
        path = self._path_for(session.session)
        serialized = session.model_dump(mode="json")
        path.write_text(
            yaml.safe_dump(serialized, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    def append_prompt(self, session: SessionFile,
                      prompt_context_id: ContextHashID) -> SessionFile:
        next_turn = SessionTurn(
            index=len(session.turns) + 1,
            prompt_id=prompt_context_id,
            response_id=None,
        )
        updated = session.model_copy(
            update={
                "turns": [*session.turns, next_turn],
                "message_ids": [*session.message_ids, prompt_context_id],
            })
        self.save(updated)
        return updated

    def attach_response_to_last_turn(self, session: SessionFile,
                                     response_context_id: ContextHashID) -> SessionFile:
        if not session.turns:
            raise RuntimeError("Cannot attach response: session has no turns.")
        turns = list(session.turns)
        turns[-1] = turns[-1].model_copy(update={"response_id": response_context_id})
        updated = session.model_copy(update={
            "turns": turns,
            "message_ids": [*session.message_ids, response_context_id],
        })
        self.save(updated)
        return updated

    def remove_last_turn(self,
                         session: SessionFile) -> tuple[SessionFile, SessionTurn | None]:
        if not session.turns:
            return session, None
        turns = list(session.turns)
        removed_turn = turns.pop()
        message_ids = [
            cid for cid in session.message_ids if cid.md5 != removed_turn.prompt_id.md5
        ]
        if removed_turn.response_id is not None:
            message_ids = [
                cid for cid in message_ids if cid.md5 != removed_turn.response_id.md5
            ]
        updated = session.model_copy(update={"turns": turns, "message_ids": message_ids})
        self.save(updated)
        return updated, removed_turn

    def save_existing(self, session: SessionFile) -> SessionFile:
        self.save(session)
        return session
