"""
EmailTriageEnv — OpenEnv-compliant environment.

Public API:
    reset(task_id)  → Observation
    step(action)    → StepResult
    state()         → EnvironmentState
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from data.emails import TASK_DATASETS
from env.graders import GRADERS, apply_null_action_penalty
from env.models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
)
from env.tasks import TASKS


class EmailTriageEnv:
    """
    Stateful environment that presents emails one by one.
    The agent must act on each email according to the active task.
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._emails: List[Dict[str, Any]] = []
        self._step: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._history: List[Dict[str, Any]] = []

    # ─── OpenEnv interface ───────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_spam") -> Observation:
        """
        Start a new episode for the given task.
        Returns the first observation.
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
            )

        self._task_id = task_id
        self._emails = TASK_DATASETS[task_id]
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Submit an action for the current email.
        Returns the next observation, reward, done flag, and info dict.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )
        if self._task_id is None:
            raise RuntimeError("Call reset() first.")

        current_record = self._emails[self._step]
        ground_truth = current_record["ground_truth"]
        email = current_record["email"]

        # Grade the action
        grader = GRADERS[self._task_id]
        reward = grader(action, ground_truth)

        # Penalise fully empty actions
        if self._is_null_action(action):
            reward = apply_null_action_penalty(reward)

        # Record history
        self._history.append(
            {
                "step": self._step,
                "email_id": email.id,
                "action": action.model_dump(exclude_none=True),
                "reward": reward.value,
                "feedback": reward.feedback,
            }
        )
        self._cumulative_reward += reward.value

        # Advance
        self._step += 1
        done = self._step >= len(self._emails)
        self._done = done

        next_obs = (
            self._make_done_observation() if done else self._make_observation()
        )

        info: Dict[str, Any] = {
            "email_id": email.id,
            "ground_truth": ground_truth,
            "cumulative_reward": self._cumulative_reward,
        }
        if done:
            info["episode_mean_reward"] = self._cumulative_reward / len(self._emails)

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return the full internal state snapshot."""
        return EnvironmentState(
            task_id=self._task_id or "",
            step_number=self._step,
            total_steps=len(self._emails),
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            history=self._history,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        task = TASKS[self._task_id]
        record = self._emails[self._step]
        return Observation(
            email=record["email"],
            task_id=self._task_id,
            task_description=task.description,
            step_number=self._step,
            total_steps=len(self._emails),
            done=False,
        )

    def _make_done_observation(self) -> Observation:
        """Sentinel observation returned when the episode is finished."""
        last_email = self._emails[-1]["email"]
        task = TASKS[self._task_id]
        return Observation(
            email=last_email,
            task_id=self._task_id,
            task_description=task.description,
            step_number=self._step,
            total_steps=len(self._emails),
            done=True,
        )

    @staticmethod
    def _is_null_action(action: Action) -> bool:
        return all(
            v is None
            for v in [
                action.spam_label,
                action.priority,
                action.department,
                action.action_items,
                action.reply_draft,
            ]
        )
