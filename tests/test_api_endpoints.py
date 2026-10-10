"""
tests/test_api_endpoints.py

Unit and integration tests for FastAPI HTTP routes and WebSocket live streaming endpoint.
Mocks the PipelineCoordinator and external services to avoid any real network or microphone calls.
"""

import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.main import app, set_coordinator
from src.auth import SESSION_COOKIE_NAME, create_session, clear_all_sessions


class TestApiEndpoints(unittest.TestCase):

    def setUp(self):
        # Create a mock coordinator with event listener tracking
        self.mock_coordinator = MagicMock()
        self.mock_coordinator.is_running = False
        self.mock_coordinator._audio_capture = MagicMock()
        self.mock_coordinator._audio_capture.device_name = "Mock Microphone"

        self.listeners = []
        self.mock_coordinator.add_event_listener.side_effect = lambda fn: self.listeners.append(fn)
        self.mock_coordinator.remove_event_listener.side_effect = lambda fn: self.listeners.remove(fn) if fn in self.listeners else None

        def fake_emit(ev):
            for fn in list(self.listeners):
                fn(ev)

        self.mock_coordinator.emit_event.side_effect = fake_emit

        # Inject mock coordinator into FastAPI application
        set_coordinator(self.mock_coordinator)
        self.client = TestClient(app)
        # Authenticate client with a valid demo session
        self.session = create_session("boomika")
        self.client.cookies.set(SESSION_COOKIE_NAME, self.session.session_id)

    def tearDown(self):
        clear_all_sessions()
        set_coordinator(None)

    # --------------------------------------------------------------------------
    # 1. HTTP Endpoint Tests
    # --------------------------------------------------------------------------
    def test_get_root(self):
        """Verifies GET / returns valid JSON status payload with endpoints."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("Phase 2", data["phase"])
        self.assertIn("endpoints", data)
        self.assertEqual(data["endpoints"]["websocket"], "/ws/live")

    def test_get_health(self):
        """Verifies GET /health returns valid health and configuration status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("chromadb_initialized", data)
        self.assertIn("indexed_chunks", data)
        self.assertIn("pipeline_streaming", data)

    def test_get_dashboard_authenticated(self):
        """Verifies GET /dashboard returns dashboard HTML when authenticated."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_get_dashboard_unauthenticated_redirects_to_login(self):
        """Verifies unauthenticated GET /dashboard redirects to /login."""
        unauth_client = TestClient(app)
        response = unauth_client.get("/dashboard", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login")

    # --------------------------------------------------------------------------
    # 2. WebSocket Connection & Event Forwarding Tests
    # --------------------------------------------------------------------------
    def test_websocket_accepts_connection_and_receives_system_status(self):
        """Verifies WebSocket /ws/live accepts client and sends initial system_status."""
        with self.client.websocket_connect("/ws/live") as ws:
            data = ws.receive_json()
            self.assertEqual(data["type"], "system_status")
            self.assertEqual(data["status"], "connected")
            self.assertFalse(data["is_running"])
            self.assertTrue(data["is_primary"])
            self.assertEqual(data["device"], "Mock Microphone")

    def test_coordinator_events_forwarded_through_websocket(self):
        """Verifies coordinator events registered via listener are forwarded to websocket."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "custom_event",
                "payload": "test_data"
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "custom_event")
            self.assertEqual(event["payload"], "test_data")

    def test_forward_transcript_partial(self):
        """Verifies transcript_partial events are forwarded through the WebSocket."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            # Simulate coordinator emitting transcript_partial
            self.mock_coordinator.emit_event({
                "type": "transcript_partial",
                "speaker": "Customer",
                "text": "The WP-400 is flashing E-102"
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "transcript_partial")
            self.assertEqual(event["speaker"], "Customer")
            self.assertEqual(event["text"], "The WP-400 is flashing E-102")

    def test_forward_transcript_final(self):
        """Verifies transcript_final events are forwarded through the WebSocket."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "transcript_final",
                "speaker": "Customer",
                "text": "The WP-400 is flashing E-102 and stopped.",
                "confidence": 0.98
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "transcript_final")
            self.assertEqual(event["text"], "The WP-400 is flashing E-102 and stopped.")
            self.assertEqual(event["confidence"], 0.98)

    def test_forward_sentiment_update(self):
        """Verifies sentiment_update events are forwarded correctly."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "sentiment_update",
                "sentiment": "Agitated",
                "confidence": 0.95,
                "rationale": "High urgency and frustration expressed."
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "sentiment_update")
            self.assertEqual(event["sentiment"], "Agitated")
            self.assertEqual(event["confidence"], 0.95)

    def test_forward_category_update(self):
        """Verifies category_update events are forwarded correctly."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "category_update",
                "category": "Technical Troubleshooting",
                "confidence": 0.92,
                "rationale": "Inquiring about alarm E-102."
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "category_update")
            self.assertEqual(event["category"], "Technical Troubleshooting")

    def test_forward_suggestion_update_with_sources(self):
        """Verifies suggestion_update events including source references are forwarded."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "suggestion_update",
                "suggestion": "Check the fan breaker and replace cartridge WP4-FL-DE1.",
                "confidence": 1.0,
                "is_grounded": True,
                "sources": [
                    {
                        "document_reference": "TS-WP400-3.1",
                        "page": 6,
                        "section": "Section 3.1"
                    }
                ],
                "category": "Technical Troubleshooting"
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "suggestion_update")
            self.assertIn("WP4-FL-DE1", event["suggestion"])
            self.assertTrue(event["is_grounded"])
            self.assertEqual(len(event["sources"]), 1)
            self.assertEqual(event["sources"][0]["document_reference"], "TS-WP400-3.1")

    def test_forward_system_error(self):
        """Verifies system_error events are forwarded through the WebSocket."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            self.mock_coordinator.emit_event({
                "type": "system_error",
                "stage": "rag",
                "message": "ChromaDB connection timeout"
            })

            event = ws.receive_json()
            self.assertEqual(event["type"], "system_error")
            self.assertEqual(event["stage"], "rag")
            self.assertIn("ChromaDB connection timeout", event["message"])

    # --------------------------------------------------------------------------
    # 3. Client Commands & Session Lifecycle Tests
    # --------------------------------------------------------------------------
    def test_start_command_handled(self):
        """Verifies client sending start command calls coordinator.start()."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            ws.send_json({"command": "start"})
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "system_status")
            self.assertEqual(resp["status"], "started")
            self.mock_coordinator.start.assert_called_once()

    def test_stop_command_handled(self):
        """Verifies client sending stop command calls coordinator.stop()."""
        self.mock_coordinator.is_running = True
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # Consume initial status

            ws.send_json({"command": "stop"})
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "system_status")
            self.assertEqual(resp["status"], "stopped")
            self.mock_coordinator.stop.assert_called_once()

    def test_disconnect_performs_cleanup(self):
        """Verifies that client disconnect unregisters the listener and stops coordinator."""
        self.mock_coordinator.is_running = True

        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_json()
            self.assertEqual(len(self.listeners), 1)

        # After context exit, client disconnected
        self.assertEqual(len(self.listeners), 0)
        self.mock_coordinator.stop.assert_called_once()

    def test_second_connection_does_not_start_duplicate_session(self):
        """Verifies second connecting client does not initiate duplicate microphone stream."""
        self.mock_coordinator.is_running = True

        with self.client.websocket_connect("/ws/live") as ws1:
            init1 = ws1.receive_json()
            self.assertTrue(init1["is_primary"])

            with self.client.websocket_connect("/ws/live") as ws2:
                init2 = ws2.receive_json()
                self.assertFalse(init2["is_primary"])
                self.assertTrue(init2["is_running"])

                # Client 2 sending start when already running does not call start() again
                ws2.send_json({"command": "start"})
                resp2 = ws2.receive_json()
                self.assertEqual(resp2["type"], "system_status")
                self.assertEqual(resp2["status"], "already_running")
                self.mock_coordinator.start.assert_not_called()

    def test_unauthenticated_websocket_rejected(self):
        """Verifies unauthenticated WebSocket request without cookie is rejected."""
        unauth_client = TestClient(app)
        with self.assertRaises(Exception):
            with unauth_client.websocket_connect("/ws/live") as ws:
                pass


if __name__ == "__main__":
    unittest.main()
