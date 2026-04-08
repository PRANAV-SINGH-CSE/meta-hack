# 📧 Email Triage — OpenEnv

A **real-world email triage environment** where AI agents learn to classify,
prioritise, route, and respond to emails — the same cognitive workflow knowledge
workers perform dozens of times a day.

Built to the [OpenEnv](https://openenv.dev) specification: typed Pydantic models,
a clean `step() / reset() / state()` HTTP API, three graded tasks with
deterministic reward functions, and a baseline inference script.

---

## Why Email Triage?

Email triage is one of the most universal and high-value knowledge-work tasks:

- **Universal**: Every professional triages email. The skill generalises to
  customer support, legal review, HR operations, and more.
- **Graded difficulty**: Spam detection (binary) → priority routing (multi-label)
  → full triage with action extraction and reply drafting.
- **Rich reward signal**: Partial credit for adjacent priority, keyword coverage
  in action items, and reply quality — providing dense feedback across the whole
  trajectory rather than a single terminal score.

---

## Observation Space

```json
{
  "email": {
    "id":            "t1_001",
    "subject":       "URGENT: You've won $5,000,000!!!",
    "body":          "...",
    "sender":        "Prize Department",
    "sender_email":  "prizes@win-now.biz",
    "timestamp":     "2024-03-01T09:00:00Z",
    "thread_length": 1
  },
  "task_id":          "task1_spam",
  "task_description": "Classify each incoming email as 'spam' or 'not_spam'...",
  "step_number":      0,
  "total_steps":      10,
  "done":             false
}
```

The agent sees **one email at a time**. `step_number` and `total_steps` let it
track episode progress.

---

## Action Space

A single union action object — populate only the fields required by the active task.

| Field          | Type                                                      | Tasks        |
|----------------|-----------------------------------------------------------|--------------|
| `spam_label`   | `"spam"` \| `"not_spam"`                                  | Task 1       |
| `priority`     | `"low"` \| `"medium"` \| `"high"` \| `"urgent"`          | Tasks 2 & 3  |
| `department`   | `"sales"` \| `"support"` \| `"billing"` \| `"hr"` \| `"technical"` \| `"general"` | Tasks 2 & 3 |
| `action_items` | `string[]` (max 10)                                       | Task 3       |
| `reply_draft`  | `string` (max 2000 chars)                                 | Task 3       |

---

## Tasks

### Task 1 — Spam vs. Not-Spam Classification *(Easy)*

**10 emails**, mix of obvious spam and legitimate workplace emails.

The agent labels each email `spam` or `not_spam`.

**Reward**: `1.0` for correct label, `0.0` for wrong. Binary.

```json
{ "spam_label": "spam" }
```

---

### Task 2 — Priority & Department Routing *(Medium)*

**10 emails**, ranging from low-priority feature requests to critical outages
and urgent HR incidents.

The agent assigns:
- a **priority level** (`low / medium / high / urgent`)
- a **department** (`sales / support / billing / hr / technical / general`)

**Reward** (per email, weights):

| Component  | Weight | Scoring                                       |
|------------|--------|-----------------------------------------------|
| priority   | 50%    | 1.0 exact · 0.5 off-by-one · 0.0 otherwise   |
| department | 50%    | 1.0 exact · 0.0 otherwise                    |

```json
{ "priority": "urgent", "department": "technical" }
```

---

### Task 3 — Full Email Triage *(Hard)*

**5 emails**, all complex multi-stakeholder situations (breaches, chargebacks,
enterprise deals, HR complaints).

The agent must:
1. Assign priority
2. Route to department
3. List concrete **action items** the recipient must take
4. Draft a brief **professional reply**

**Reward** (per email, weights):

| Component    | Weight | Scoring                                                |
|--------------|--------|--------------------------------------------------------|
| priority     | 20%    | 1.0 exact · 0.5 off-by-one · 0.0 otherwise            |
| department   | 20%    | 1.0 exact · 0.0 otherwise                             |
| action_items | 35%    | Fraction of required keywords found (partial credit)  |
| reply_draft  | 25%    | Keyword coverage + small length bonus                 |

```json
{
  "priority":     "urgent",
  "department":   "technical",
  "action_items": [
    "Approve gateway restart immediately",
    "Notify enterprise clients of ongoing incident",
    "Escalate to VP Engineering / CTO",
    "Monitor SLA timer — breach in 40 minutes"
  ],
  "reply_draft": "We have escalated this issue urgently to our on-call engineering team and authorised the gateway restart. We are actively monitoring the SLA timer and will contact your enterprise clients directly within the next 10 minutes..."
}
```

---

## Reward Design

Every task returns a **dense per-step reward** in `[0.0, 1.0]`:

- **Partial credit** prevents cliff edges — an agent that routes to the right
  department but picks `high` instead of `urgent` still earns 0.75.
- **Keyword coverage** for action items rewards the agent for identifying the
  right set of concrete next steps, even if phrasing differs.
- **Reply quality** is assessed by keyword presence and minimum length, penalising
  trivially short or empty drafts.
- **Null action penalty** (`−0.3`) discourages the agent from skipping emails.

---

## API Reference

The server exposes a simple HTTP API. Full interactive docs at `/docs`.

| Method | Path     | Body / Params       | Returns           |
|--------|----------|---------------------|-------------------|
| GET    | /health  | —                   | `{status: "ok"}` |
| GET    | /tasks   | —                   | Task metadata     |
| POST   | /reset   | `{task_id: string}` | `Observation`     |
| POST   | /step    | `Action`            | `StepResult`      |
| GET    | /state   | —                   | `EnvironmentState`|

### StepResult schema

```json
{
  "observation": { ...Observation },
  "reward": {
    "value": 0.75,
    "breakdown": { "priority": 0.5, "department": 1.0 },
    "feedback": "priority: ~ (got 'high', expected 'urgent') | department: ✓"
  },
  "done": false,
  "info": {
    "email_id": "t2_001",
    "cumulative_reward": 0.75
  }
}
```

---

## Setup & Usage

### Option A — Docker (recommended)

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Verify
curl http://localhost:7860/health
```

### Option B — Local Python

```bash
# Install
pip install -r requirements.txt

# Run server
python app.py

# In another shell — run baseline
OPENAI_API_KEY=sk-... python baseline.py
```

### Running the baseline

```bash
OPENAI_API_KEY=sk-...  python baseline.py --model gpt-4o-mini

# Single task
OPENAI_API_KEY=sk-...  python baseline.py --task task3_full_triage

# Against a custom model/provider
OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
  OPENAI_API_KEY=gsk_... \
  python baseline.py --model llama-3.1-70b-versatile
```

### Validate with OpenEnv CLI

```bash
pip install openenv-cli
openenv validate --config openenv.yaml --url http://localhost:7860
```

---

## Baseline Scores

Measured with **gpt-4o-mini** at temperature 0:

| Task                       | Difficulty | Steps | Mean Reward |
|----------------------------|------------|-------|-------------|
| `task1_spam`               | Easy       | 10    | **0.9000**  |
| `task2_routing`            | Medium     | 10    | **0.7250**  |
| `task3_full_triage`        | Hard       | 5     | **0.6120**  |
| **Overall**                |            | **25**| **0.7457**  |

**Error analysis**:
- Task 1: One phishing email misclassified as not-spam (trusted-looking sender domain).
- Task 2: Two off-by-one priority errors at the `medium/high` boundary (ambiguous urgency language).
- Task 3: Action items miss nuanced legal deadlines; reply drafts sometimes omit required escalation keywords.

---

## Project Structure

```
email-triage-env/
├── app.py                  # FastAPI server (OpenEnv HTTP API)
├── baseline.py             # LLM agent baseline (OpenAI-compatible)
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container for HF Spaces
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── environment.py      # EmailTriageEnv class (reset/step/state)
│   ├── models.py           # Pydantic models (Observation, Action, Reward, ...)
│   ├── graders.py          # Per-task graders with partial-credit logic
│   └── tasks.py            # Task registry & metadata
├── data/
│   ├── __init__.py
│   └── emails.py           # 25 synthetic emails with ground-truth annotations
└── tests/
    └── test_environment.py # 14 pytest tests (perfect/zero/edge-case agents)
```

---

## Extending the Environment

**Add a new task:**
1. Add email records with `ground_truth` to `data/emails.py`
2. Register a `TaskMeta` in `env/tasks.py`
3. Write a grader function in `env/graders.py` and add it to `GRADERS`
4. Update `openenv.yaml`

**Add more emails per task:** The environment iterates `TASK_DATASETS[task_id]`
— just append records.

**Custom grader hooks:** Override `EmailTriageEnv` and call `super().step()`,
then post-process the `Reward` object.

---

## License

MIT
