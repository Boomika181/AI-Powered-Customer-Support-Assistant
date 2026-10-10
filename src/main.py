"""
src/main.py

FastAPI Application Entry Point.
Phase 0/1: Health and configuration checks.
Phase 2: WebSocket live streaming (/ws/live) and dashboard support.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline_coordinator import PipelineCoordinator
from src.auth import (
    SESSION_COOKIE_NAME,
    DEFAULT_SESSION_LIFETIME_SECONDS,
    authenticate_user,
    register_user,
    create_session,
    get_session,
    delete_session,
)

logger = logging.getLogger("api")

app = FastAPI(
    title="AI-Powered Customer Support Assistant (POC)",
    description="Localhost Proof of Concept - Phase 2 Backend with Authentication",
    version="0.2.0"
)

# Static directory mount
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not any(getattr(route, "name", None) == "static" for route in app.routes):
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    confirm_password: str


# ------------------------------------------------------------------------------
# Coordinator Singleton & Session Management
# ------------------------------------------------------------------------------
_coordinator: Optional[PipelineCoordinator] = None
_active_websockets: Set[WebSocket] = set()
_session_lock = asyncio.Lock()


def get_coordinator() -> PipelineCoordinator:
    """Returns the active PipelineCoordinator instance, creating one if needed."""
    global _coordinator
    if _coordinator is None:
        _coordinator = PipelineCoordinator()
    return _coordinator


def set_coordinator(coordinator: Optional[PipelineCoordinator]) -> None:
    """Injects or resets the active PipelineCoordinator (used for testing)."""
    global _coordinator
    _coordinator = coordinator


# ------------------------------------------------------------------------------
# HTTP Routes
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "phase": "Phase 2 (FastAPI & Dashboard Integration)",
        "message": "AI-Powered Customer Support Assistant POC backend is running.",
        "endpoints": {
            "health": "/health",
            "websocket": "/ws/live",
            "dashboard": "/dashboard",
            "login": "/login",
            "auth_login": "/api/auth/login",
            "auth_register": "/api/auth/register",
            "auth_logout": "/api/auth/logout",
            "auth_me": "/api/auth/me",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    import chromadb

    chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    col_name = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")

    db_ok = False
    chunk_count = 0
    if chroma_dir.exists():
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            col = client.get_collection(col_name)
            chunk_count = col.count()
            db_ok = chunk_count > 0
        except Exception:
            db_ok = False

    coord = _coordinator
    is_streaming = coord.is_running if coord else False

    return {
        "status": "healthy" if db_ok else "setup_required",
        "chromadb_initialized": db_ok,
        "indexed_chunks": chunk_count,
        "assemblyai_configured": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "pipeline_streaming": is_streaming,
        "active_clients": len(_active_websockets)
    }


# ------------------------------------------------------------------------------
# Authentication Routes
# ------------------------------------------------------------------------------
@app.get("/login")
def get_login(request: Request):
    """
    Serves the login page. Redirects authenticated agents directly to /dashboard.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id and get_session(session_id):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    login_file = STATIC_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "login.html not found"}
    )


@app.post("/api/auth/login")
async def api_login(credentials: LoginRequest, request: Request, response: Response):
    """
    Validates agent credentials and sets an HttpOnly session cookie.
    Does NOT leak password, salt, hash, or session ID in the response body.
    """
    email_or_username = credentials.email.strip() if credentials.email else ""
    password = credentials.password if credentials.password else ""

    if not email_or_username or not password:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Email and password are required."}
        )

    user = authenticate_user(email_or_username, password)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"authenticated": False, "error": "Invalid email or password."}
        )

    session = create_session(user["username"])
    is_secure = request.url.scheme == "https"

    resp = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "authenticated": True,
            "agent_name": session.agent_name,
            "email": session.email,
        }
    )
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
        max_age=DEFAULT_SESSION_LIFETIME_SECONDS,
    )
    return resp


@app.post("/api/auth/register")
async def api_register(data: RegisterRequest, request: Request, response: Response):
    """
    Registers a new support agent and establishes an active authenticated session.
    """
    name = data.name.strip() if data.name else ""
    email = data.email.strip() if data.email else ""
    password = data.password if data.password else ""
    confirm_password = data.confirm_password if data.confirm_password else ""

    if not name:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Agent full name is required."}
        )
    if not email:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Work email address is required."}
        )
    if not password:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Password is required."}
        )
    if len(password) < 10:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Password must be at least 10 characters long."}
        )
    if password != confirm_password:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": "Passwords do not match."}
        )

    try:
        user_record = register_user(name, email, password)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"authenticated": False, "error": str(exc)}
        )
    except Exception as exc:
        logger.error("Registration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"authenticated": False, "error": "An error occurred during account creation."}
        )

    session = create_session(user_record["username"])
    is_secure = request.url.scheme == "https"

    resp = JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "authenticated": True,
            "agent_name": session.agent_name,
            "email": session.email,
            "message": "Account created successfully."
        }
    )
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
        max_age=DEFAULT_SESSION_LIFETIME_SECONDS,
    )
    return resp


@app.post("/api/auth/logout")
async def api_logout(request: Request, response: Response):
    """
    Invalidates current agent session on the server and clears the session cookie.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        delete_session(session_id)

    resp = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"authenticated": False, "message": "Logged out successfully."}
    )
    resp.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return resp


@app.get("/api/auth/me")
def api_auth_me(request: Request):
    """
    Returns the authenticated support agent's identity.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    session = get_session(session_id) if session_id else None
    if not session:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"authenticated": False, "error": "Not authenticated"}
        )
    return {
        "authenticated": True,
        "username": session.username,
        "agent_name": session.agent_name,
        "email": session.email,
    }


@app.get("/dashboard")
def get_dashboard(request: Request):
    """
    Protected dashboard endpoint. Unauthenticated users are redirected to /login.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    session = get_session(session_id) if session_id else None
    if not session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "pending",
        "message": "Dashboard UI not yet created."
    }


# ------------------------------------------------------------------------------
# WebSocket Live Endpoint
# ------------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    session_id = websocket.cookies.get(SESSION_COOKIE_NAME)
    session = get_session(session_id) if session_id else None
    if not session:
        await websocket.close(code=1008, reason="Unauthorized: Valid agent session required")
        return

    await websocket.accept()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    coordinator = get_coordinator()

    # Thread-safe event listener bridging coordinator events to async queue
    def on_coordinator_event(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    coordinator.add_event_listener(on_coordinator_event)

    async with _session_lock:
        is_first_client = len(_active_websockets) == 0
        _active_websockets.add(websocket)

    # Initial connection status payload
    initial_status = {
        "type": "system_status",
        "status": "connected",
        "is_running": coordinator.is_running,
        "is_primary": is_first_client,
        "device": getattr(getattr(coordinator, "_audio_capture", None), "device_name", "Default Microphone"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket.send_json(initial_status)

    forwarder_task = asyncio.create_task(_forward_events(websocket, event_queue))

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "start":
                if not coordinator.is_running:
                    coordinator.start()
                    await websocket.send_json({
                        "type": "system_status",
                        "status": "started",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "system_status",
                        "status": "already_running",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

            elif command == "stop":
                if coordinator.is_running:
                    coordinator.stop()
                    await websocket.send_json({
                        "type": "system_status",
                        "status": "stopped",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "system_status",
                        "status": "already_stopped",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("Error in WebSocket session loop: %s", exc)
    finally:
        # Cleanup listener
        coordinator.remove_event_listener(on_coordinator_event)

        async with _session_lock:
            _active_websockets.discard(websocket)
            remaining_clients = len(_active_websockets)

        # Stop coordinator if no clients remain
        if remaining_clients == 0 and coordinator.is_running:
            try:
                coordinator.stop()
            except Exception as e:
                logger.warning("Error stopping coordinator on disconnect: %s", e)

        # Terminate event forwarder
        event_queue.put_nowait(None)
        await forwarder_task


async def _forward_events(websocket: WebSocket, event_queue: asyncio.Queue[Optional[Dict[str, Any]]]) -> None:
    """Continuously consumes events from the async queue and pushes JSON to the client."""
    try:
        while True:
            event = await event_queue.get()
            if event is None:
                break
            await websocket.send_json(event)
    except Exception as exc:
        logger.debug("Event forwarder terminated: %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=5000, reload=True)
