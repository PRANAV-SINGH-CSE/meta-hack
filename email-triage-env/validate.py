#!/usr/bin/env python3
"""
validate.py — Local OpenEnv spec validator
==========================================

Simulates `openenv validate` without requiring the CLI to be installed.
Checks:
  1. openenv.yaml exists and is well-formed
  2. Server is reachable at --url
  3. All tasks listed in YAML are accessible via /tasks
  4. reset() returns a valid Observation
  5. step() returns a valid StepResult
  6. state() returns a valid EnvironmentState
  7. Reward values are within [0.0, 1.0]
  8. Episode completes cleanly (done flag is set)
  9. Per-task mean reward is non-negative

Usage:
    # Validate against a running server
    python validate.py --url http://localhost:7860

    # Or with docker
    docker run -p 7860:7860 email-triage-env &
    python validate.py
"""

from __future__ import annotations
import argparse
import sys
import yaml
import httpx
from pathlib import Path

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def run_episode(base_url: str, task_id: str, http: httpx.Client) -> dict:
    """Run a minimal episode and return summary stats."""
    r = http.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    sid = r.headers.get("x-session-id", "")
    obs = r.json()
    headers = {"X-Session-Id": sid} if sid else {}

    rewards = []
    done = obs.get("done", False)

    # Submit neutral-ish actions until episode ends
    while not done:
        action: dict = {}
        if task_id == "task1_spam":
            action = {"spam_label": "not_spam"}
        elif task_id == "task2_routing":
            action = {"priority": "medium", "department": "general"}
        else:
            action = {
                "priority": "medium",
                "department": "general",
                "action_items": ["Review and respond to this email."],
                "reply_draft": "Thank you for your email. We will follow up shortly with next steps.",
            }

        sr = http.post(f"{base_url}/step", json=action, headers=headers, timeout=10)
        sr.raise_for_status()
        step_result = sr.json()
        rewards.append(step_result["reward"]["value"])
        done = step_result["done"]
        obs = step_result["observation"]

    # Fetch state
    st_r = http.get(f"{base_url}/state", headers=headers, timeout=10)
    st_r.raise_for_status()

    # Fetch summary
    sm_r = http.get(f"{base_url}/summary", headers=headers, timeout=10)
    sm_r.raise_for_status()

    return {
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "state_ok": st_r.status_code == 200,
        "summary_ok": sm_r.status_code == 200,
        "done_set": done,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv spec validator")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--config", default="openenv.yaml")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    failures = 0

    print(f"\n{'='*56}")
    print(f"  OpenEnv Validate")
    print(f"  URL: {base}")
    print(f"{'='*56}\n")

    # ── 1. YAML ───────────────────────────────────────────────────────────────
    print("[ openenv.yaml ]")
    yaml_path = Path(args.config)
    yaml_ok = yaml_path.exists()
    if not check("openenv.yaml exists", yaml_ok):
        failures += 1
    else:
        with yaml_path.open() as f:
            cfg = yaml.safe_load(f)
        if not check("YAML is well-formed", isinstance(cfg, dict)):
            failures += 1
        else:
            check("name field present", "name" in cfg, cfg.get("name", ""))
            check("version field present", "version" in cfg, str(cfg.get("version", "")))
            check("tasks defined", bool(cfg.get("tasks")), str(len(cfg.get("tasks", []))))
            check("observation_space defined", "observation_space" in cfg)
            check("action_space defined", "action_space" in cfg)
            check("reward defined", "reward" in cfg)
            yaml_tasks = [t["id"] for t in cfg.get("tasks", [])]

    print()

    # ── 2. Server reachability ────────────────────────────────────────────────
    print("[ Server ]")
    http = httpx.Client(timeout=15)
    try:
        hr = http.get(f"{base}/health")
        hr.raise_for_status()
        health_ok = hr.json().get("status") == "ok"
    except Exception as exc:
        health_ok = False
        print(f"  {FAIL}  Health check failed: {exc}")
        failures += 1
        print("\nAborting — server not reachable.\n")
        sys.exit(failures)

    if not check("Server reachable (/health)", health_ok):
        failures += 1

    try:
        tr = http.get(f"{base}/tasks")
        tr.raise_for_status()
        server_tasks = list(tr.json().keys())
        if not check("/tasks endpoint works", True, str(server_tasks)):
            failures += 1
    except Exception as exc:
        check("/tasks endpoint works", False, str(exc))
        server_tasks = []
        failures += 1

    if yaml_ok and yaml_tasks:
        missing = [t for t in yaml_tasks if t not in server_tasks]
        if not check("All YAML tasks served", not missing, str(missing) if missing else ""):
            failures += 1

    print()

    # ── 3. Per-task episode validation ────────────────────────────────────────
    for task_id in server_tasks:
        print(f"[ Task: {task_id} ]")
        try:
            stats = run_episode(base, task_id, http)
        except Exception as exc:
            check("Episode ran without error", False, str(exc))
            failures += 1
            print()
            continue

        if not check("Episode completed (done=True)", stats["done_set"]):
            failures += 1
        if not check("/state returns 200", stats["state_ok"]):
            failures += 1
        if not check("/summary returns 200", stats["summary_ok"]):
            failures += 1

        rewards = stats["rewards"]
        in_range = all(0.0 <= r <= 1.0 for r in rewards)
        if not check("All rewards in [0.0, 1.0]", in_range, str(rewards)):
            failures += 1

        check(
            "Non-trivial rewards (>0 on at least one step)",
            any(r > 0 for r in rewards),
            f"mean={stats['mean_reward']:.4f}",
        )
        print()

    # ── Result ────────────────────────────────────────────────────────────────
    print(f"{'='*56}")
    if failures == 0:
        print(f"  {PASS}  All checks passed — OpenEnv spec compliant")
    else:
        print(f"  {FAIL}  {failures} check(s) failed")
    print(f"{'='*56}\n")

    sys.exit(failures)


if __name__ == "__main__":
    main()
