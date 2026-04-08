"""
env/session.py — Thread-safe session manager.

Allows multiple independent agents to run concurrently against the same
server, each with their own isolated environment instance.

Sessions are created on /reset and identified by a UUID token returned
in the response's `info` dict and the `X-Session-Id` response header.
Idle sessions are evicted after SESSION_TTL_SECONDS.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, Optional, Tuple

from env.environment import EmailTriageEnv
from env.models import EnvironmentState, Observation, StepResult
from env.models import Action

SESSION_TTL_SECONDS = 3600          # 1 hour idle timeout
MAX_SESSIONS = 256                  # hard cap to prevent memory exhaustion


class Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.env = EmailTriageEnv()
        self.lock = threading.Lock()
        self.last_active = time.monotonic()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active


class SessionManager:
    """Holds all active sessions and evicts stale ones."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._global_lock = threading.Lock()
        self._start_eviction_thread()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        with self._global_lock:
            if len(self._sessions) >= MAX_SESSIONS:
                self._evict_oldest()
            sid = str(uuid.uuid4())
            self._sessions[sid] = Session(sid)
            return sid

    def get(self, session_id: str) -> Optional[Session]:
        with self._global_lock:
            s = self._sessions.get(session_id)
            if s:
                s.touch()
            return s

    def delete(self, session_id: str) -> bool:
        with self._global_lock:
            return self._sessions.pop(session_id, None) is not None

    def active_count(self) -> int:
        with self._global_lock:
            return len(self._sessions)

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_stale(self) -> int:
        now = time.monotonic()
        stale = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active) > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_active)
        del self._sessions[oldest.session_id]

    def _start_eviction_thread(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(300)          # run every 5 minutes
                with self._global_lock:
                    self._evict_stale()

        t = threading.Thread(target=_loop, daemon=True)
        t.start()


# Singleton — imported by app.py
session_manager = SessionManager()
