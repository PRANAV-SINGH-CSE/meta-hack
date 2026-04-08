#!/usr/bin/env python3
"""
baseline.py — Baseline inference script for Email Triage OpenEnv
=================================================================

Runs an LLM (via OpenAI-compatible API) against all three tasks and
reports per-task and overall scores.

Usage:
    OPENAI_API_KEY=<key> python baseline.py [--model gpt-4o-mini] [--base-url URL]

Environment variables:
    OPENAI_API_KEY   — required
    OPENAI_BASE_URL  — optional, defaults to OpenAI (set for other providers)
    ENV_BASE_URL     — base URL of the running EmailTriageEnv API
                       default: http://localhost:7860
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENV_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = ["task1_spam", "task2_routing", "task3_full_triage"]

# ─── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant.

You will be given an email and a task description. Your job is to analyse the
email carefully and respond with a JSON object containing ONLY the action fields
required for the current task. Do not include any prose — output raw JSON only.

Action schema (use only the fields relevant to the task):
{
  "spam_label":   "spam" | "not_spam",
  "priority":     "low" | "medium" | "high" | "urgent",
  "department":   "sales" | "support" | "billing" | "hr" | "technical" | "general",
  "action_items": ["<concrete action>", ...],   // list of strings, task 3 only
  "reply_draft":  "<professional reply text>"   // task 3 only
}

Guidelines:
- spam_label: used ONLY for task 1
- priority + department: used for tasks 2 and 3
- action_items: list 3–6 concrete, specific actions the recipient must take
- reply_draft: 60–150 word professional reply acknowledging the email and outlining next steps

Output ONLY valid JSON. No markdown fences. No explanation.
"""


# ─── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str = DEFAULT_ENV_URL) -> None:
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=30)

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self.http.post(f"{self.base}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.http.post(f"{self.base}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.http.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> Dict[str, Any]:
        r = self.http.get(f"{self.base}/tasks")
        r.raise_for_status()
        return r.json()


# ─── Agent loop ───────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any]) -> str:
    email = obs["email"]
    return f"""TASK: {obs['task_description']}

EMAIL ({obs['step_number'] + 1}/{obs['total_steps']}):
  From:    {email['sender']} <{email['sender_email']}>
  Subject: {email['subject']}
  Date:    {email['timestamp']}
  Thread:  {email['thread_length']} message(s)

Body:
{email['body']}

Respond with a JSON action object now."""


def call_llm(client: OpenAI, model: str, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wraps in them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def run_task(
    task_id: str,
    env: EnvClient,
    llm: OpenAI,
    model: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}")
        print(f"{'='*60}")

    obs = env.reset(task_id)
    rewards: List[float] = []
    feedbacks: List[str] = []

    while not obs.get("done", False):
        user_prompt = build_user_prompt(obs)
        try:
            action = call_llm(llm, model, user_prompt)
        except Exception as exc:
            print(f"  [LLM ERROR] {exc} — submitting null action")
            action = {}

        result = env.step(action)
        reward_val = result["reward"]["value"]
        feedback = result["reward"]["feedback"]
        rewards.append(reward_val)
        feedbacks.append(feedback)

        if verbose:
            email_id = obs["email"]["id"]
            print(f"  [{email_id}] reward={reward_val:.3f} | {feedback}")

        obs = result["observation"]
        time.sleep(0.3)   # polite rate limiting

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    if verbose:
        print(f"\n  → Mean reward: {mean_reward:.4f}  ({len(rewards)} steps)")

    return {
        "task_id": task_id,
        "steps": len(rewards),
        "mean_reward": mean_reward,
        "total_reward": sum(rewards),
        "per_step_rewards": rewards,
        "per_step_feedbacks": feedbacks,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL)
    parser.add_argument("--task", default=None, help="Run a single task (optional)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL")
    llm = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    env = EnvClient(args.env_url)

    # Verify env is reachable
    try:
        tasks_meta = env.tasks()
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {args.env_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    task_ids = [args.task] if args.task else TASK_IDS
    results = []

    for task_id in task_ids:
        if task_id not in tasks_meta:
            print(f"Unknown task '{task_id}', skipping.")
            continue
        result = run_task(task_id, env, llm, args.model, verbose=not args.quiet)
        results.append(result)

    # ─── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    print(f"  {'Task':<30} {'Steps':>5} {'Mean Reward':>12} {'Total':>8}")
    print(f"  {'-'*57}")
    overall_rewards = []
    for r in results:
        print(
            f"  {r['task_id']:<30} {r['steps']:>5} "
            f"{r['mean_reward']:>12.4f} {r['total_reward']:>8.3f}"
        )
        overall_rewards.extend(r["per_step_rewards"])
    print(f"  {'-'*57}")
    if overall_rewards:
        grand_mean = sum(overall_rewards) / len(overall_rewards)
        print(f"  {'OVERALL':<30} {len(overall_rewards):>5} {grand_mean:>12.4f}")
    print(f"{'='*60}\n")

    # Machine-readable output
    output = {
        "model": args.model,
        "tasks": results,
        "overall_mean_reward": grand_mean if overall_rewards else 0.0,
    }
    print("JSON OUTPUT:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
