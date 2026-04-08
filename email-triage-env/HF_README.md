---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - nlp
  - email
  - real-world
  - agent
license: mit
short_description: Real-world email triage environment for AI agents (OpenEnv spec)
---

# Email Triage — OpenEnv

> A real-world email triage environment. Three tasks of increasing difficulty.
> Full OpenEnv spec compliance: typed models, step/reset/state API, dense rewards.

## Quick start

```bash
# Reset to task 1
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_spam"}'

# Submit an action
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"spam_label": "spam"}'
```

See the full [README](README.md) for docs, task descriptions, and baseline scores.
