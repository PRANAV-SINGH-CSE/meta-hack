#!/usr/bin/env python3
"""
demo_agent.py — Rule-based baseline agent (no API key required)
===============================================================

Demonstrates the full environment loop using a deterministic
rule-based agent. Useful for:
  - Verifying the environment runs end-to-end
  - Establishing a rule-based lower-bound score
  - CI smoke tests

Usage:
    python demo_agent.py                    # all tasks
    python demo_agent.py --task task1_spam  # single task
    python demo_agent.py --env-url http://localhost:7860
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_ENV_URL = "http://localhost:7860"

# ─── Rule-based classifiers ───────────────────────────────────────────────────

SPAM_SIGNALS = [
    r"won.*\$", r"\$\d+.*million", r"click.*claim", r"verify.*account",
    r"nigerian prince", r"90% off", r"make.*\$.*month", r"work from home",
    r"limited time", r"act now", r"bank detail", r"prize.*department",
    r"luxury.*deal", r"account.*comprom", r"suspended.*\d+ hour",
]

DEPT_SIGNALS = {
    "technical": [r"server", r"database", r"bug", r"api", r"crash", r"outage",
                  r"error", r"down", r"503", r"500", r"deploy", r"security",
                  r"breach", r"s3", r"gdpr", r"infosec"],
    "billing":   [r"invoice", r"payment", r"charge", r"refund", r"billing",
                  r"subscription", r"expire", r"chargeback", r"funds"],
    "sales":     [r"enterprise", r"deal", r"demo", r"proposal", r"seat",
                  r"partnership", r"opportunity", r"budget", r"vendor"],
    "hr":        [r"onboard", r"performance review", r"expense", r"harassment",
                  r"confidential", r"employee", r"hr", r"leave", r"lms"],
    "support":   [r"feature request", r"dark mode", r"feedback", r"suggestion",
                  r"mobile app"],
}

URGENCY_SIGNALS = {
    "urgent": [r"critical", r"urgent", r"now", r"immediately", r"sla breach",
               r"down\b", r"outage", r"expir.*tomorrow", r"18 hour",
               r"24 hour", r"gdpr.*notify", r"harassment"],
    "high":   [r"asap", r"today", r"this week", r"enterprise", r"chargeback",
               r"affected.*customer", r"3 customer", r"penalty"],
    "medium": [r"next week", r"please review", r"partnership", r"proposal",
               r"schedule"],
}


def _match(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def classify_spam(subject: str, body: str, sender_email: str) -> str:
    text = f"{subject} {body} {sender_email}"
    return "spam" if _match(text, SPAM_SIGNALS) else "not_spam"


def classify_department(subject: str, body: str) -> str:
    text = f"{subject} {body}"
    for dept, patterns in DEPT_SIGNALS.items():
        if _match(text, patterns):
            return dept
    return "general"


def classify_priority(subject: str, body: str) -> str:
    text = f"{subject} {body}"
    for level in ["urgent", "high", "medium"]:
        if _match(text, URGENCY_SIGNALS[level]):
            return level
    return "low"


def extract_action_items(subject: str, body: str) -> List[str]:
    """Heuristic: pull sentences that contain imperative verbs."""
    imperative_re = re.compile(
        r"(?:please\s+)?(?:need|must|require|ensure|submit|notify|contact|"
        r"gather|escalate|approve|schedule|review|confirm|document|secure)\b.{5,80}",
        re.IGNORECASE,
    )
    text = f"{subject}. {body}"
    items = imperative_re.findall(text)
    # Deduplicate and cap
    seen, result = set(), []
    for item in items:
        key = item.strip().lower()[:40]
        if key not in seen:
            seen.add(key)
            result.append(item.strip().rstrip(".") + ".")
        if len(result) >= 5:
            break
    if not result:
        result = ["Review email and take appropriate action."]
    return result


def draft_reply(subject: str, priority: str, department: str) -> str:
    urgency_phrase = {
        "urgent": "We are treating this as a critical priority and will respond immediately.",
        "high":   "We recognise the time sensitivity and will address this today.",
        "medium": "We will review this and follow up within 1-2 business days.",
        "low":    "Thank you for reaching out. We will follow up shortly.",
    }[priority]

    return (
        f"Thank you for your email regarding '{subject}'.\n\n"
        f"{urgency_phrase} "
        f"Your message has been routed to our {department} team who will "
        f"handle your request.\n\n"
        f"We will keep you updated as we progress. Please do not hesitate "
        f"to reply to this email if you have additional information or questions.\n\n"
        f"Best regards,\nEmail Triage Team"
    )


# ─── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=15)
        self._session_id: Optional[str] = None

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self.http.post(f"{self.base}/reset", json={"task_id": task_id})
        r.raise_for_status()
        self._session_id = r.headers.get("x-session-id")
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        headers = {}
        if self._session_id:
            headers["X-Session-Id"] = self._session_id
        r = self.http.post(f"{self.base}/step", json=action, headers=headers)
        r.raise_for_status()
        return r.json()

    def summary(self) -> Dict[str, Any]:
        headers = {}
        if self._session_id:
            headers["X-Session-Id"] = self._session_id
        r = self.http.get(f"{self.base}/summary", headers=headers)
        r.raise_for_status()
        return r.json()


# ─── Agent decision ────────────────────────────────────────────────────────────

def decide(task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    email = obs["email"]
    subj = email["subject"]
    body = email["body"]
    sender = email["sender_email"]

    if task_id == "task1_spam":
        return {"spam_label": classify_spam(subj, body, sender)}

    priority = classify_priority(subj, body)
    department = classify_department(subj, body)

    if task_id == "task2_routing":
        return {"priority": priority, "department": department}

    # task3_full_triage
    return {
        "priority": priority,
        "department": department,
        "action_items": extract_action_items(subj, body),
        "reply_draft": draft_reply(subj, priority, department),
    }


# ─── Runner ────────────────────────────────────────────────────────────────────

def run_task(task_id: str, env: EnvClient, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(f"\n{'='*60}\n  TASK: {task_id}\n{'='*60}")

    obs = env.reset(task_id)
    rewards: List[float] = []

    while not obs.get("done", False):
        action = decide(task_id, obs)
        result = env.step(action)
        r = result["reward"]["value"]
        rewards.append(r)
        if verbose:
            eid = obs["email"]["id"]
            fb = result["reward"]["feedback"]
            print(f"  [{eid}] reward={r:.3f} | {fb}")
        obs = result["observation"]

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    if verbose:
        print(f"\n  → Mean reward: {mean_r:.4f}  ({len(rewards)} steps)")

    return {
        "task_id": task_id,
        "steps": len(rewards),
        "mean_reward": mean_r,
        "total_reward": sum(rewards),
        "per_step_rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based demo agent for Email Triage OpenEnv")
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL)
    parser.add_argument("--task", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = EnvClient(args.env_url)

    try:
        r = httpx.get(f"{args.env_url}/health", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach {args.env_url} — {exc}", file=sys.stderr)
        sys.exit(1)

    task_ids = [args.task] if args.task else ["task1_spam", "task2_routing", "task3_full_triage"]
    results = []

    for tid in task_ids:
        results.append(run_task(tid, env, verbose=not args.quiet))

    print(f"\n{'='*60}")
    print("  RULE-BASED AGENT — RESULTS")
    print(f"{'='*60}")
    print(f"  {'Task':<30} {'Steps':>5} {'Mean Reward':>12}")
    print(f"  {'-'*49}")
    all_rewards: List[float] = []
    for r in results:
        print(f"  {r['task_id']:<30} {r['steps']:>5} {r['mean_reward']:>12.4f}")
        all_rewards.extend(r["per_step_rewards"])
    print(f"  {'-'*49}")
    grand = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"  {'OVERALL':<30} {len(all_rewards):>5} {grand:>12.4f}")
    print(f"{'='*60}\n")
    print(json.dumps({"agent": "rule_based", "tasks": results, "overall_mean": grand}, indent=2))


if __name__ == "__main__":
    main()
