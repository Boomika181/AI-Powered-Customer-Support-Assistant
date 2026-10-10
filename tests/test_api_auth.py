"""
tests/test_api_auth.py

Comprehensive tests for authentication HTTP endpoints, session cookies,
dashboard route protection, and WebSocket authentication enforcement.
"""

import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.main import app, set_coordinator
from src.auth import (
    SESSION_COOKIE_NAME,
    get_session,
    clear_all_sessions,
    reset_demo_registry,
)


class TestApiAuthentication(unittest.TestCase):

    def setUp(self):
        clear_all_sessions()
        reset_demo_registry()
        self.mock_coordinator = MagicMock()
        self.mock_coordinator.is_running = False
        self.mock_coordinator._audio_capture = MagicMock()
        self.mock_coordinator._audio_capture.device_name = "Mock Mic"
        set_coordinator(self.mock_coordinator)
        self.client = TestClient(app)

    def tearDown(self):
        clear_all_sessions()
        reset_demo_registry()
        set_coordinator(None)

    # --------------------------------------------------------------------------
    # 1. Login Endpoint (POST /api/auth/login)
    # --------------------------------------------------------------------------
    def test_login_valid_boomika_email(self):
        """Verifies valid login with boomika's email returns 200, sets cookie, displays Boomika."""
        response = self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["agent_name"], "Boomika")
        self.assertEqual(data["email"], "boomika@support.industrial.ai")
        # Ensure security: No secrets or session IDs in response body
        self.assertNotIn("session_id", data)
        self.assertNotIn("password", data)
        self.assertNotIn("salt", data)
        self.assertNotIn("hash", data)

        # Cookie must be set
        self.assertIn(SESSION_COOKIE_NAME, response.cookies)
        session_id = response.cookies[SESSION_COOKIE_NAME]
        self.assertIsNotNone(get_session(session_id))

    def test_login_valid_alex_email(self):
        """Verifies valid login with alex's email returns 200, sets cookie, displays Alex Chen."""
        response = self.client.post("/api/auth/login", json={
            "email": "alex@support.industrial.ai",
            "password": "Machinery#400"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["agent_name"], "Alex Chen")
        self.assertEqual(data["email"], "alex@support.industrial.ai")
        self.assertIn(SESSION_COOKIE_NAME, response.cookies)

    def test_login_valid_username_backward_compatibility(self):
        """Verifies login also succeeds with plain username for backward compatibility."""
        response = self.client.post("/api/auth/login", json={
            "email": "boomika",
            "password": "Industrial@2026"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["agent_name"], "Boomika")

    def test_login_invalid_password(self):
        """Verifies login with wrong password returns 401 and does not set cookie."""
        response = self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "WrongPassword123"
        })
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["authenticated"])
        self.assertIn("error", data)
        self.assertNotIn(SESSION_COOKIE_NAME, response.cookies)

    def test_login_unknown_email(self):
        """Verifies login with non-existent email returns 401."""
        response = self.client.post("/api/auth/login", json={
            "email": "nonexistent@industrial.ai",
            "password": "Industrial@2026"
        })
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["authenticated"])

    def test_login_missing_fields(self):
        """Verifies empty or missing fields return 400."""
        res1 = self.client.post("/api/auth/login", json={"email": "", "password": "abc"})
        self.assertEqual(res1.status_code, 400)

        res2 = self.client.post("/api/auth/login", json={"email": "boomika", "password": ""})
        self.assertEqual(res2.status_code, 400)

    # --------------------------------------------------------------------------
    # 2. Identity Verification (GET /api/auth/me)
    # --------------------------------------------------------------------------
    def test_auth_me_authenticated(self):
        """Verifies /api/auth/me returns the identity of the logged-in agent."""
        self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["username"], "boomika")
        self.assertEqual(data["agent_name"], "Boomika")
        self.assertEqual(data["email"], "boomika@support.industrial.ai")

    def test_auth_me_unauthenticated(self):
        """Verifies /api/auth/me returns 401 when no session cookie is provided."""
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["authenticated"])

    # --------------------------------------------------------------------------
    # 3. Logout Endpoint (POST /api/auth/logout)
    # --------------------------------------------------------------------------
    def test_logout_invalidates_session_and_clears_cookie(self):
        """Verifies logout deletes server-side session and clears cookie."""
        login_resp = self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        session_id = login_resp.cookies[SESSION_COOKIE_NAME]
        self.assertIsNotNone(get_session(session_id))

        logout_resp = self.client.post("/api/auth/logout")
        self.assertEqual(logout_resp.status_code, 200)
        self.assertFalse(logout_resp.json()["authenticated"])

        # Server-side session must be invalidated
        self.assertIsNone(get_session(session_id))

        # Subsequent call to /api/auth/me must fail with 401
        me_resp = self.client.get("/api/auth/me")
        self.assertEqual(me_resp.status_code, 401)

    # --------------------------------------------------------------------------
    # 4. Route Protection (GET /dashboard, GET /login)
    # --------------------------------------------------------------------------
    def test_dashboard_redirects_unauthenticated_user(self):
        """Verifies visiting /dashboard without valid session redirects to /login."""
        response = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login")

    def test_dashboard_allows_authenticated_user(self):
        """Verifies visiting /dashboard with valid session returns 200 HTML."""
        self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        response = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_login_page_redirects_already_authenticated_user(self):
        """Verifies visiting /login when already authenticated redirects to /dashboard."""
        self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        response = self.client.get("/login", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/dashboard")

    # --------------------------------------------------------------------------
    # 5. WebSocket Authentication Protection (/ws/live)
    # --------------------------------------------------------------------------
    def test_websocket_accepts_authenticated_session(self):
        """Verifies WebSocket accepts connection when client has valid session cookie."""
        self.client.post("/api/auth/login", json={
            "email": "boomika@support.industrial.ai",
            "password": "Industrial@2026"
        })
        with self.client.websocket_connect("/ws/live") as ws:
            status_msg = ws.receive_json()
            self.assertEqual(status_msg["type"], "system_status")
            self.assertEqual(status_msg["status"], "connected")

    def test_websocket_rejects_unauthenticated_connection(self):
        """Verifies WebSocket rejects connection when no session cookie is present."""
        unauth_client = TestClient(app)
        with self.assertRaises(Exception):
            with unauth_client.websocket_connect("/ws/live") as ws:
                pass

    # --------------------------------------------------------------------------
    # 6. Registration Endpoint (POST /api/auth/register)
    # --------------------------------------------------------------------------
    def test_register_success_creates_session_and_cookie(self):
        """Verifies valid registration returns 201, sets session cookie, returns agent name."""
        response = self.client.post("/api/auth/register", json={
            "name": "Marcus Wright",
            "email": "marcus@support.industrial.ai",
            "password": "Resistance#2026",
            "confirm_password": "Resistance#2026"
        })
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["agent_name"], "Marcus Wright")
        self.assertEqual(data["email"], "marcus@support.industrial.ai")
        # Ensure security: No secret leaks
        self.assertNotIn("session_id", data)
        self.assertNotIn("password", data)
        self.assertNotIn("salt", data)
        self.assertNotIn("hash", data)

        # Cookie must be set
        self.assertIn(SESSION_COOKIE_NAME, response.cookies)
        session_id = response.cookies[SESSION_COOKIE_NAME]
        session = get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.agent_name, "Marcus Wright")

    def test_register_missing_fields_returns_400(self):
        """Verifies missing required fields in registration return 400."""
        res1 = self.client.post("/api/auth/register", json={
            "name": "",
            "email": "test@industrial.ai",
            "password": "ValidPassword123",
            "confirm_password": "ValidPassword123"
        })
        self.assertEqual(res1.status_code, 400)

        res2 = self.client.post("/api/auth/register", json={
            "name": "Marcus",
            "email": "",
            "password": "ValidPassword123",
            "confirm_password": "ValidPassword123"
        })
        self.assertEqual(res2.status_code, 400)

    def test_register_password_too_short_returns_400(self):
        """Verifies password shorter than 10 characters returns 400."""
        response = self.client.post("/api/auth/register", json={
            "name": "Marcus Wright",
            "email": "marcus@support.industrial.ai",
            "password": "Short9ch",
            "confirm_password": "Short9ch"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("10 characters", response.json()["error"])

    def test_register_password_mismatch_returns_400(self):
        """Verifies mismatched passwords return 400."""
        response = self.client.post("/api/auth/register", json={
            "name": "Marcus Wright",
            "email": "marcus@support.industrial.ai",
            "password": "ValidPassword123",
            "confirm_password": "DifferentPassword123"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("match", response.json()["error"].lower())

    def test_register_duplicate_email_returns_400(self):
        """Verifies registering with existing email returns 400."""
        # Try registering with boomika's email
        response = self.client.post("/api/auth/register", json={
            "name": "Duplicate Boomika",
            "email": "boomika@support.industrial.ai",
            "password": "ValidPassword123",
            "confirm_password": "ValidPassword123"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("already exists", response.json()["error"].lower())

    def test_registered_user_can_sign_in_and_access_dashboard(self):
        """Verifies newly registered user can sign in via /api/auth/login and access dashboard."""
        reg_resp = self.client.post("/api/auth/register", json={
            "name": "Kyle Reese",
            "email": "kyle@support.industrial.ai",
            "password": "TechCom#2026",
            "confirm_password": "TechCom#2026"
        })
        self.assertEqual(reg_resp.status_code, 201)

        # Clear session cookies to simulate new login
        self.client.cookies.clear()

        # Login with newly registered credentials
        login_resp = self.client.post("/api/auth/login", json={
            "email": "kyle@support.industrial.ai",
            "password": "TechCom#2026"
        })
        self.assertEqual(login_resp.status_code, 200)
        self.assertEqual(login_resp.json()["agent_name"], "Kyle Reese")

        # Identity via /api/auth/me returns Kyle Reese
        me_resp = self.client.get("/api/auth/me")
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.json()["agent_name"], "Kyle Reese")

        # Can access /dashboard
        dash_resp = self.client.get("/dashboard")
        self.assertEqual(dash_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
