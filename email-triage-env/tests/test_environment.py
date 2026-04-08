"""
tests/test_environment.py
Runs all three tasks end-to-end with perfect and zero agents to verify scoring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import EmailTriageEnv
from env.models import Action
from data.emails import TASK1_EMAILS, TASK2_EMAILS, TASK3_EMAILS, TASK_DATASETS


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return EmailTriageEnv()


# ─── Task 1: Spam Classification ─────────────────────────────────────────────

class TestTask1Spam:
    def test_perfect_agent_scores_1(self, env):
        obs = env.reset("task1_spam")
        total = 0.0
        steps = 0
        while not obs.done:
            gt = TASK1_EMAILS[obs.step_number]["ground_truth"]
            result = env.step(Action(spam_label=gt["spam_label"]))
            total += result.reward.value
            steps += 1
            obs = result.observation
        assert steps == 10
        assert total / steps == pytest.approx(1.0)

    def test_zero_agent_scores_0(self, env):
        obs = env.reset("task1_spam")
        while not obs.done:
            # Deliberately wrong label for all
            wrong = "spam" if TASK1_EMAILS[obs.step_number]["ground_truth"]["spam_label"] == "not_spam" else "not_spam"
            result = env.step(Action(spam_label=wrong))
            assert result.reward.value == pytest.approx(0.0)
            obs = result.observation

    def test_null_action_penalised(self, env):
        env.reset("task1_spam")
        result = env.step(Action())  # all None
        assert result.reward.value < 0.5  # penalty applied

    def test_done_flag_raises_on_step(self, env):
        obs = env.reset("task1_spam")
        for i in range(10):
            gt = TASK1_EMAILS[i]["ground_truth"]
            result = env.step(Action(spam_label=gt["spam_label"]))
            obs = result.observation
        assert obs.done
        with pytest.raises(RuntimeError, match="done"):
            env.step(Action(spam_label="spam"))


# ─── Task 2: Priority Routing ─────────────────────────────────────────────────

class TestTask2Routing:
    def test_perfect_agent_scores_1(self, env):
        obs = env.reset("task2_routing")
        rewards = []
        while not obs.done:
            gt = TASK2_EMAILS[obs.step_number]["ground_truth"]
            result = env.step(Action(priority=gt["priority"], department=gt["department"]))
            rewards.append(result.reward.value)
            obs = result.observation
        assert len(rewards) == 10
        assert sum(rewards) / len(rewards) == pytest.approx(1.0)

    def test_adjacent_priority_gets_half_credit(self, env):
        """urgent is off-by-one from high → 0.5 priority score."""
        env.reset("task2_routing")
        # First email: urgent + technical
        result = env.step(Action(priority="high", department="technical"))
        # priority score 0.5, department 1.0 → 0.5*0.5 + 0.5*1.0 = 0.75
        assert result.reward.breakdown["priority"] == pytest.approx(0.5)
        assert result.reward.breakdown["department"] == pytest.approx(1.0)
        assert result.reward.value == pytest.approx(0.75)

    def test_wrong_department_no_credit(self, env):
        env.reset("task2_routing")
        result = env.step(Action(priority="urgent", department="billing"))
        assert result.reward.breakdown["department"] == pytest.approx(0.0)


# ─── Task 3: Full Triage ──────────────────────────────────────────────────────

class TestTask3FullTriage:
    def _make_perfect_action(self, step_num: int) -> Action:
        from data.emails import TASK3_EMAILS
        record = TASK3_EMAILS[step_num]
        gt = record["ground_truth"]
        # Build action with all required fields perfectly filled
        action_items = [item for item in gt["required_action_items"]]
        # Build a reply containing all required keywords
        reply = "We acknowledge this email. " + " ".join(gt["required_reply_keywords"]) + \
                ". We will take immediate action and follow up shortly. " * 3
        return Action(
            priority=gt["priority"],
            department=gt["department"],
            action_items=action_items,
            reply_draft=reply,
        )

    def test_perfect_agent_scores_above_0_9(self, env):
        obs = env.reset("task3_full_triage")
        rewards = []
        while not obs.done:
            action = self._make_perfect_action(obs.step_number)
            result = env.step(action)
            rewards.append(result.reward.value)
            obs = result.observation
        assert len(rewards) == 5
        assert sum(rewards) / len(rewards) >= 0.9

    def test_missing_action_items_reduces_score(self, env):
        obs = env.reset("task3_full_triage")
        gt = TASK3_EMAILS[0]["ground_truth"]
        # No action items, no reply draft
        result = env.step(Action(
            priority=gt["priority"],
            department=gt["department"],
        ))
        assert result.reward.breakdown["action_items"] == pytest.approx(0.0)
        assert result.reward.breakdown["reply_draft"] == pytest.approx(0.0)
        # Max possible: 0.2 + 0.2 = 0.4
        assert result.reward.value <= 0.42

    def test_short_reply_draft_penalised(self, env):
        env.reset("task3_full_triage")
        gt = TASK3_EMAILS[0]["ground_truth"]
        result = env.step(Action(
            priority=gt["priority"],
            department=gt["department"],
            action_items=["approve restart"],
            reply_draft="OK",  # too short
        ))
        assert result.reward.breakdown["reply_draft"] == pytest.approx(0.0)


# ─── State & reset ────────────────────────────────────────────────────────────

class TestStateAndReset:
    def test_state_tracks_cumulative_reward(self, env):
        env.reset("task1_spam")
        env.step(Action(spam_label="spam"))   # correct (email 0 is spam)
        s = env.state()
        assert s.step_number == 1
        assert s.cumulative_reward == pytest.approx(1.0)

    def test_reset_clears_state(self, env):
        env.reset("task1_spam")
        env.step(Action(spam_label="spam"))
        env.reset("task2_routing")
        s = env.state()
        assert s.step_number == 0
        assert s.cumulative_reward == pytest.approx(0.0)
        assert s.task_id == "task2_routing"

    def test_unknown_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset("task99_nonexistent")

    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(Action(spam_label="spam"))
