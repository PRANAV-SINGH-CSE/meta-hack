"""
Email Triage — OpenEnv HTTP API  (multi-session + built-in UI)
================================================================

Endpoints
---------
GET  /                    interactive browser UI
GET  /health              liveness probe
GET  /tasks               list task metadata
POST /reset               start episode  → X-Session-Id header
POST /step                submit action  ← X-Session-Id header
GET  /state               current state  ← X-Session-Id header
GET  /summary             episode summary
DELETE /session           close session
GET  /sessions            active session count (admin)
GET  /docs                OpenAPI / Swagger
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Action, EnvironmentState, Observation, StepResult
from env.session import session_manager, Session
from env.tasks import TASKS

UI_FILE = Path(__file__).parent / "ui.html"
DEFAULT_SESSION_ID = "__default__"


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_manager._sessions[DEFAULT_SESSION_ID] = Session(DEFAULT_SESSION_ID)
    yield


app = FastAPI(
    title="Email Triage — OpenEnv",
    description=(
        "Real-world email triage environment implementing the OpenEnv spec.\n\n"
        "Three tasks of increasing difficulty: spam classification, priority routing, "
        "and full triage with action-item extraction and reply drafting.\n\n"
        "Supports concurrent agents via session tokens (X-Session-Id header)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


class ResetRequest(BaseModel):
    task_id: str = "task1_spam"


def _resolve_session(session_id: Optional[str]):
    sid = session_id or DEFAULT_SESSION_ID
    s = session_manager.get(sid)
    if s is None:
        raise HTTPException(404, f"Session '{sid}' not found. Call POST /reset first.")
    return s


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    if UI_FILE.exists():
        return FileResponse(UI_FILE, media_type="text/html")
    return {"message": "Email Triage OpenEnv API — see /docs"}


# ── Meta ──────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    return {"status": "ok", "active_sessions": session_manager.active_count()}


@app.get("/tasks", tags=["meta"])
async def list_tasks() -> Dict[str, Any]:
    return {tid: t.model_dump() for tid, t in TASKS.items()}


@app.get("/sessions", tags=["meta"])
async def active_sessions() -> Dict[str, Any]:
    return {"active_sessions": session_manager.active_count()}


# ── OpenEnv core ──────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, tags=["openenv"])
async def reset(req: ResetRequest, response: Response) -> Observation:
    sid = session_manager.create_session()
    s = session_manager.get(sid)
    try:
        with s.lock:
            obs = s.env.reset(req.task_id)
    except ValueError as exc:
        session_manager.delete(sid)
        raise HTTPException(400, str(exc))
    response.headers["X-Session-Id"] = sid
    return obs


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(
    action: Action,
    x_session_id: Optional[str] = Header(default=None),
) -> StepResult:
    s = _resolve_session(x_session_id)
    try:
        with s.lock:
            result = s.env.step(action)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    return result


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
async def state(
    x_session_id: Optional[str] = Header(default=None),
) -> EnvironmentState:
    s = _resolve_session(x_session_id)
    with s.lock:
        return s.env.state()


@app.get("/summary", tags=["openenv"])
async def summary(
    x_session_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    s = _resolve_session(x_session_id)
    with s.lock:
        st = s.env.state()
    steps = len(st.history)
    rewards = [h["reward"] for h in st.history]
    feedbacks = [h["feedback"] for h in st.history]
    mean_r = sum(rewards) / steps if steps else 0.0
    return {
        "session_id": s.session_id,
        "task_id": st.task_id,
        "steps_taken": steps,
        "total_steps": st.total_steps,
        "done": st.done,
        "total_reward": st.cumulative_reward,
        "mean_reward": round(mean_r, 4),
        "per_step_rewards": rewards,
        "per_step_feedbacks": feedbacks,
        "history": st.history,
    }


@app.delete("/session", tags=["meta"])
async def close_session(
    x_session_id: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    sid = x_session_id or DEFAULT_SESSION_ID
    if not session_manager.delete(sid):
        raise HTTPException(404, f"Session '{sid}' not found.")
    return {"deleted": sid}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
