"""
OpenEnv typed models — Email Triage Environment
All request/response shapes are Pydantic-validated.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Domain objects ───────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    sender_email: str
    timestamp: str
    thread_length: int = 1                  # how many prior messages exist


class TaskMeta(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int                          # emails the agent must process


# ─── OpenEnv core models ───────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees after reset() or step()."""
    email: Email
    task_id: str
    task_description: str
    step_number: int                        # 0-indexed current email index
    total_steps: int                        # total emails in this task episode
    done: bool


class Action(BaseModel):
    """
    Union action space — fields used depend on the active task.

    Task 1 (easy)   — spam_label only.
    Task 2 (medium) — priority + department.
    Task 3 (hard)   — priority + department + action_items + reply_draft.
    """
    # Task 1
    spam_label: Optional[Literal["spam", "not_spam"]] = None

    # Task 2 + 3
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    department: Optional[Literal[
        "sales", "support", "billing", "hr", "technical", "general"
    ]] = None

    # Task 3 only
    action_items: Optional[List[str]] = Field(default=None, max_length=10)
    reply_draft: Optional[str] = Field(default=None, max_length=2000)


class Reward(BaseModel):
    """Granular reward with partial-credit breakdown."""
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]             # component → score
    feedback: str                           # human-readable explanation


class StepResult(BaseModel):
    """Return value of step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeSummary(BaseModel):
    """Returned by state() when done=True, or via /summary endpoint."""
    task_id: str
    steps_taken: int
    total_reward: float
    mean_reward: float
    per_step_rewards: List[float]
    per_step_feedback: List[str]


class EnvironmentState(BaseModel):
    """Full internal snapshot — returned by state()."""
    task_id: str
    step_number: int
    total_steps: int
    done: bool
    cumulative_reward: float
    history: List[Dict[str, Any]]           # list of {action, reward, email_id}
