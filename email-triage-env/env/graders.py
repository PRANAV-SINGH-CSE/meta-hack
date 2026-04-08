"""
Agent graders for all three tasks.

Each grader takes (action, ground_truth) and returns a Reward.
All scores are in [0.0, 1.0].
"""

from __future__ import annotations
from typing import Any, Dict, List
from env.models import Action, Reward

# Priority ladder for adjacent-credit scoring
_PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


def _priority_score(predicted: str | None, expected: str) -> float:
    """
    Full credit for exact match.
    Half credit if off by exactly one step.
    Zero otherwise.
    """
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0
    if abs(_PRIORITY_ORDER.get(predicted, -99) - _PRIORITY_ORDER[expected]) == 1:
        return 0.5
    return 0.0


def _department_score(predicted: str | None, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def _action_items_score(items: List[str] | None, required: List[str]) -> float:
    """
    Partial credit: fraction of required action-item keywords found
    in the agent's action_items list (case-insensitive substring match).
    """
    if not items:
        return 0.0
    combined = " ".join(items).lower()
    matched = sum(1 for kw in required if kw.lower() in combined)
    return matched / len(required)


def _reply_draft_score(draft: str | None, required_keywords: List[str]) -> float:
    """
    Partial credit: fraction of required keywords present in the reply draft.
    Also penalises an empty/trivially short draft.
    """
    if not draft or len(draft.strip()) < 20:
        return 0.0
    draft_lower = draft.lower()
    matched = sum(1 for kw in required_keywords if kw.lower() in draft_lower)
    base = matched / len(required_keywords)
    # Small length bonus — encourages substantive replies
    length_bonus = min(0.1, len(draft.split()) / 500)
    return min(1.0, base + length_bonus)


# ─── Task graders ──────────────────────────────────────────────────────────────

def grade_task1(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 1 — Spam Classification (easy).
    Binary: correct label = 1.0, wrong = 0.0.
    """
    expected = ground_truth["spam_label"]
    correct = action.spam_label == expected
    score = 1.0 if correct else 0.0

    return Reward(
        value=score,
        breakdown={"spam_label": score},
        feedback=(
            f"✓ Correct — '{expected}'" if correct
            else f"✗ Wrong — predicted '{action.spam_label}', expected '{expected}'"
        ),
    )


def grade_task2(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 2 — Priority + Department Routing (medium).

    Weights:
      priority   50%  (with adjacent-step partial credit)
      department 50%  (exact match)
    """
    expected_priority = ground_truth["priority"]
    expected_dept = ground_truth["department"]

    p_score = _priority_score(action.priority, expected_priority)
    d_score = _department_score(action.department, expected_dept)

    total = 0.5 * p_score + 0.5 * d_score

    parts: List[str] = []
    parts.append(
        f"priority: {'✓' if p_score == 1.0 else ('~' if p_score > 0 else '✗')} "
        f"(got '{action.priority}', expected '{expected_priority}')"
    )
    parts.append(
        f"department: {'✓' if d_score == 1.0 else '✗'} "
        f"(got '{action.department}', expected '{expected_dept}')"
    )

    return Reward(
        value=round(total, 4),
        breakdown={"priority": p_score, "department": d_score},
        feedback=" | ".join(parts),
    )


def grade_task3(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 3 — Full Triage (hard).

    Weights:
      priority      20%  (adjacent-step partial credit)
      department    20%  (exact match)
      action_items  35%  (keyword-coverage partial credit)
      reply_draft   25%  (keyword-coverage partial credit)
    """
    expected_priority = ground_truth["priority"]
    expected_dept = ground_truth["department"]
    required_items = ground_truth.get("required_action_items", [])
    required_keywords = ground_truth.get("required_reply_keywords", [])

    p_score = _priority_score(action.priority, expected_priority)
    d_score = _department_score(action.department, expected_dept)
    ai_score = _action_items_score(action.action_items, required_items)
    rd_score = _reply_draft_score(action.reply_draft, required_keywords)

    total = 0.20 * p_score + 0.20 * d_score + 0.35 * ai_score + 0.25 * rd_score

    parts = [
        f"priority: {p_score:.2f} (got '{action.priority}', expected '{expected_priority}')",
        f"department: {d_score:.2f} (got '{action.department}', expected '{expected_dept}')",
        f"action_items: {ai_score:.2f} ({len(action.action_items or [])} items provided)",
        f"reply_draft: {rd_score:.2f} ({len((action.reply_draft or '').split())} words)",
    ]

    return Reward(
        value=round(total, 4),
        breakdown={
            "priority": p_score,
            "department": d_score,
            "action_items": ai_score,
            "reply_draft": rd_score,
        },
        feedback=" | ".join(parts),
    )


# ─── Penalty helpers ────────────────────────────────────────────────────────────

def apply_null_action_penalty(reward: Reward, fraction: float = 0.3) -> Reward:
    """
    Penalise an agent that submits an entirely empty / null action.
    Reduces reward by `fraction` (clamped to 0).
    """
    new_value = max(0.0, reward.value - fraction)
    return Reward(
        value=new_value,
        breakdown=reward.breakdown,
        feedback=reward.feedback + " | PENALTY: null/empty action submitted",
    )


GRADERS = {
    "task1_spam": grade_task1,
    "task2_routing": grade_task2,
    "task3_full_triage": grade_task3,
}
