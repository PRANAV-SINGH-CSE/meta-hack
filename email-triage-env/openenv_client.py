"""
openenv_client.py — Thin Python SDK for Email Triage OpenEnv
=============================================================

Wraps the HTTP API in a Pythonic interface. Can be used standalone
or imported by agent scripts.

Example
-------
from openenv_client import EmailTriageClient, Action

client = EmailTriageClient("http://localhost:7860")

# Run a full episode
obs = client.reset("task2_routing")
while not obs.done:
    result = client.step(Action(priority="high", department="technical"))
    print(result.reward.value, result.reward.feedback)
    obs = result.observation

summary = client.summary()
print(f"Mean reward: {summary['mean_reward']}")
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import httpx

# Re-export models so callers don't need to import env.models separately
from env.models import Action, Observation, StepResult, EnvironmentState


class EmailTriageClient:
    """
    Synchronous HTTP client for the Email Triage OpenEnv server.

    Parameters
    ----------
    base_url : str
        Base URL of the running server, e.g. "http://localhost:7860"
    timeout : float
        Per-request timeout in seconds (default 30)
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)
        self._session_id: Optional[str] = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_spam") -> Observation:
        """Start a new episode. Returns the first observation."""
        r = self._post("/reset", {"task_id": task_id})
        self._session_id = r.headers.get("x-session-id")
        return Observation.model_validate(r.json())

    def step(self, action: Action) -> StepResult:
        """Submit an action. Returns the next observation + reward."""
        r = self._post("/step", action.model_dump(exclude_none=True))
        return StepResult.model_validate(r.json())

    def state(self) -> EnvironmentState:
        """Return the full internal state snapshot."""
        r = self._get("/state")
        return EnvironmentState.model_validate(r.json())

    def summary(self) -> Dict[str, Any]:
        """Return the episode summary (rewards, feedbacks, history)."""
        return self._get("/summary").json()

    def tasks(self) -> Dict[str, Any]:
        """List available tasks and their metadata."""
        return self._get("/tasks").json()

    def health(self) -> Dict[str, Any]:
        """Liveness probe."""
        return self._get("/health").json()

    def close_session(self) -> None:
        """Explicitly release the current session on the server."""
        if self._session_id:
            self._delete("/session")
            self._session_id = None

    # ── Convenience ────────────────────────────────────────────────────────────

    def run_episode(
        self,
        task_id: str,
        policy_fn,              # Callable[[Observation], Action]
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a complete episode using a policy function.

        Parameters
        ----------
        task_id : str
            One of "task1_spam", "task2_routing", "task3_full_triage"
        policy_fn : Callable[[Observation], Action]
            Function that takes an Observation and returns an Action
        verbose : bool
            Print step-level rewards if True

        Returns
        -------
        dict with keys: task_id, steps, mean_reward, total_reward, per_step_rewards
        """
        obs = self.reset(task_id)
        rewards = []
        feedbacks = []

        while not obs.done:
            action = policy_fn(obs)
            result = self.step(action)
            rewards.append(result.reward.value)
            feedbacks.append(result.reward.feedback)
            if verbose:
                print(f"  step={len(rewards)} reward={result.reward.value:.3f} | {result.reward.feedback}")
            obs = result.observation

        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "task_id": task_id,
            "steps": len(rewards),
            "mean_reward": mean_r,
            "total_reward": sum(rewards),
            "per_step_rewards": rewards,
            "per_step_feedbacks": feedbacks,
        }

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "EmailTriageClient":
        return self

    def __exit__(self, *_) -> None:
        self.close_session()
        self._http.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def _session_headers(self) -> Dict[str, str]:
        if self._session_id:
            return {"X-Session-Id": self._session_id}
        return {}

    def _post(self, path: str, body: Any) -> httpx.Response:
        r = self._http.post(
            f"{self._base}{path}",
            json=body,
            headers=self._session_headers,
        )
        r.raise_for_status()
        return r

    def _get(self, path: str) -> httpx.Response:
        r = self._http.get(
            f"{self._base}{path}",
            headers=self._session_headers,
        )
        r.raise_for_status()
        return r

    def _delete(self, path: str) -> httpx.Response:
        r = self._http.request(
            "DELETE",
            f"{self._base}{path}",
            headers=self._session_headers,
        )
        r.raise_for_status()
        return r
