"""Store interface (template requirement T3). Both implementations must pass the same contract tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..engine.gateway import CommitResult, Proposal
from ..engine.model import Problem
from ..engine.pack import Pack


class Store(ABC):
    kind: str = "abstract"

    def __init__(self, pack: Pack) -> None:
        self.pack = pack

    # ---- reads -------------------------------------------------------------
    @abstractmethod
    def load_problem(self) -> Problem: ...

    @abstractmethod
    def available_columns(self) -> dict[str, set[str]]: ...

    @abstractmethod
    def get_proposal(self, plan_id: str) -> Proposal | None: ...

    @abstractmethod
    def list_proposals(self, limit: int = 10) -> list[Proposal]: ...

    @abstractmethod
    def audit_rows(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def outbox(self) -> list[dict[str, Any]]: ...

    def last_job_ids(self) -> list[str]:
        return []

    # ---- append-only writes (no operational state change) -------------------
    @abstractmethod
    def save_proposal(self, p: Proposal) -> None: ...

    @abstractmethod
    def record_approval(self, plan_id: str, plan_hash: str, approver: str, channel: str) -> dict[str, Any]: ...

    @abstractmethod
    def reject(self, plan_id: str, reason: str, actor: str) -> dict[str, Any]: ...

    # ---- the only operational mutation path -------------------------------
    @abstractmethod
    def commit(self, plan_id: str, approval_code: str, actor: str,
               approver: str | None = None) -> CommitResult:
        """approver is set only in APPROVAL_MODE=confirm, after ADK's human confirmation:
        the approval row and the commit are then written in the same atomic step."""

    # ---- ingest paths (real-world events; CLI or provider only, never agent tools) ----
    @abstractmethod
    def update_conditions(self, rows: list[dict[str, Any]], source: str, observed_at: str,
                          actor: str) -> dict[str, Any]: ...

    @abstractmethod
    def record_field_update(self, lot_id: str, delta: int, actor: str, note: str) -> dict[str, Any]: ...

    @abstractmethod
    def mark_outbox_sent(self, plan_id: str) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...
