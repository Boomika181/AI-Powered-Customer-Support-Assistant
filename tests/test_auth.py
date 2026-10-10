"""
tests/test_auth.py

Unit tests for Phase 1 Authentication and Session Management Foundation (src/auth.py).
"""

from datetime import datetime, timezone, timedelta
import unittest

from src.auth import (
    SESSION_COOKIE_NAME,
    DEFAULT_SESSION_LIFETIME_SECONDS,
    USER_REGISTRY,
    Session,
    authenticate_user,
    create_session,
    get_session,
    delete_session,
    is_session_expired,
    cleanup_expired_sessions,
    clear_all_sessions,
    hash_password,
    verify_password,
    register_user,
    reset_demo_registry,
)


class TestAuthenticationFoundation(unittest.TestCase):

    def setUp(self):
        clear_all_sessions()
        reset_demo_registry()

    def tearDown(self):
        clear_all_sessions()
        reset_demo_registry()

    # --------------------------------------------------------------------------
    # 1. User Registry & Password Storage Integrity
    # --------------------------------------------------------------------------
    def test_user_registry_contains_hashes_not_plaintext(self):
        """Verifies stored user records contain salt and hex hash, never plaintext passwords."""
        forbidden_plaintexts = ["Industrial@2026", "Machinery#400"]

        for username, record in USER_REGISTRY.items():
            self.assertIn("username", record)
            self.assertIn("agent_name", record)
            self.assertIn("salt", record)
            self.assertIn("password_hash", record)

            # Ensure salt is raw bytes
            self.assertIsInstance(record["salt"], bytes)
            self.assertGreater(len(record["salt"]), 0)

            # Ensure password_hash is a 64-char sha256 hex string
            self.assertIsInstance(record["password_hash"], str)
            self.assertEqual(len(record["password_hash"]), 64)

            # Strict check: Plaintext passwords must NOT appear anywhere in the record values
            for plain in forbidden_plaintexts:
                self.assertNotIn(plain, str(record))

    def test_hash_and_verify_password_logic(self):
        """Verifies PBKDF2 hash computation and timing-safe verification."""
        salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        pwd = "TestSecretPassword#123"
        h = hash_password(pwd, salt)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)

        # Verification succeeds with matching password
        self.assertTrue(verify_password(pwd, salt, h))
        # Verification fails with incorrect password
        self.assertFalse(verify_password("WrongPassword", salt, h))
        # Verification fails with invalid inputs
        self.assertFalse(verify_password("", salt, h))
        self.assertFalse(verify_password(pwd, b"", h))
        self.assertFalse(verify_password(pwd, salt, ""))

    # --------------------------------------------------------------------------
    # 2. Credential Verification & Authentication
    # --------------------------------------------------------------------------
    def test_authenticate_valid_credentials_boomika(self):
        """Verifies authenticating boomika returns correct agent identity."""
        identity = authenticate_user("boomika", "Industrial@2026")
        self.assertIsNotNone(identity)
        self.assertEqual(identity["username"], "boomika")
        self.assertEqual(identity["agent_name"], "Boomika")
        # Ensure identity is minimal and strictly limited to username and agent_name
        self.assertEqual(set(identity.keys()), {"username", "agent_name"})

    def test_authenticate_valid_credentials_alex(self):
        """Verifies authenticating alex returns correct agent identity."""
        identity = authenticate_user("alex", "Machinery#400")
        self.assertIsNotNone(identity)
        self.assertEqual(identity["username"], "alex")
        self.assertEqual(identity["agent_name"], "Alex Chen")
        self.assertEqual(set(identity.keys()), {"username", "agent_name"})

    def test_authenticate_username_case_insensitive(self):
        """Verifies usernames are normalized case-insensitively."""
        identity1 = authenticate_user("Boomika", "Industrial@2026")
        identity2 = authenticate_user("ALEX", "Machinery#400")
        self.assertIsNotNone(identity1)
        self.assertEqual(identity1["agent_name"], "Boomika")
        self.assertIsNotNone(identity2)
        self.assertEqual(identity2["agent_name"], "Alex Chen")

    def test_authenticate_invalid_password(self):
        """Verifies wrong password fails authentication."""
        result = authenticate_user("boomika", "WrongPassword123")
        self.assertIsNone(result)

    def test_authenticate_unknown_user(self):
        """Verifies non-existent username fails authentication."""
        result = authenticate_user("unknown_user", "Industrial@2026")
        self.assertIsNone(result)

    def test_authenticate_empty_credentials(self):
        """Verifies empty and whitespace-only credentials fail safely."""
        self.assertIsNone(authenticate_user("", "Industrial@2026"))
        self.assertIsNone(authenticate_user("boomika", ""))
        self.assertIsNone(authenticate_user("", ""))
        self.assertIsNone(authenticate_user("   ", "Industrial@2026"))

    def test_authenticate_none_credentials(self):
        """Verifies None parameters fail safely without exceptions."""
        self.assertIsNone(authenticate_user(None, "password"))
        self.assertIsNone(authenticate_user("boomika", None))
        self.assertIsNone(authenticate_user(None, None))

    # --------------------------------------------------------------------------
    # 3. Session Creation & Unique Tokens
    # --------------------------------------------------------------------------
    def test_create_session_success(self):
        """Verifies session creation stores correct agent details and timestamps."""
        session = create_session("boomika")
        self.assertIsInstance(session, Session)
        self.assertEqual(session.username, "boomika")
        self.assertEqual(session.agent_name, "Boomika")
        self.assertIsInstance(session.session_id, str)
        self.assertGreater(len(session.session_id), 30)
        self.assertIsInstance(session.created_at, datetime)
        self.assertIsInstance(session.expires_at, datetime)

        # Check default 8-hour lifetime
        expected_diff = timedelta(seconds=DEFAULT_SESSION_LIFETIME_SECONDS)
        actual_diff = session.expires_at - session.created_at
        self.assertAlmostEqual(actual_diff.total_seconds(), expected_diff.total_seconds(), delta=2.0)

    def test_create_session_unique_ids(self):
        """Verifies each generated session ID is distinct and unpredictable."""
        session_ids = {create_session("boomika").session_id for _ in range(50)}
        self.assertEqual(len(session_ids), 50)

    def test_create_session_unknown_user_raises(self):
        """Verifies create_session raises ValueError for unregistered user."""
        with self.assertRaises(ValueError):
            create_session("ghost_agent")

    def test_create_session_invalid_lifetime_raises(self):
        """Verifies create_session raises ValueError if lifetime is non-positive."""
        with self.assertRaises(ValueError):
            create_session("boomika", lifetime_seconds=0)
        with self.assertRaises(ValueError):
            create_session("boomika", lifetime_seconds=-100)

    # --------------------------------------------------------------------------
    # 4. Session Retrieval & Invalidation
    # --------------------------------------------------------------------------
    def test_get_session_valid(self):
        """Verifies retrieving an active session by session_id."""
        session = create_session("alex")
        retrieved = get_session(session.session_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, session.session_id)
        self.assertEqual(retrieved.agent_name, "Alex Chen")

    def test_get_session_invalid_ids(self):
        """Verifies get_session returns None for missing or invalid IDs."""
        self.assertIsNone(get_session("non_existent_token"))
        self.assertIsNone(get_session(""))
        self.assertIsNone(get_session(None))

    def test_delete_session(self):
        """Verifies deleting a session removes it from the store."""
        session = create_session("boomika")
        self.assertIsNotNone(get_session(session.session_id))

        deleted = delete_session(session.session_id)
        self.assertTrue(deleted)
        self.assertIsNone(get_session(session.session_id))

        # Subsequent deletion returns False
        self.assertFalse(delete_session(session.session_id))
        self.assertFalse(delete_session("arbitrary_id"))
        self.assertFalse(delete_session(""))
        self.assertFalse(delete_session(None))

    # --------------------------------------------------------------------------
    # 5. Session Expiration & Cleanup
    # --------------------------------------------------------------------------
    def test_is_session_expired(self):
        """Verifies expiration detection with custom time injection."""
        session = create_session("boomika", lifetime_seconds=60)
        now_valid = session.created_at + timedelta(seconds=30)
        now_expired = session.created_at + timedelta(seconds=61)

        self.assertFalse(is_session_expired(session, now=now_valid))
        self.assertTrue(is_session_expired(session, now=now_expired))
        self.assertFalse(session.is_expired(now=now_valid))
        self.assertTrue(session.is_expired(now=now_expired))

    def test_get_session_auto_purges_expired(self):
        """Verifies get_session automatically purges and returns None if expired."""
        session = create_session("boomika", lifetime_seconds=60)
        future_time = session.created_at + timedelta(seconds=120)

        # Retrieve at future time -> should be None
        result = get_session(session.session_id, now=future_time)
        self.assertIsNone(result)

        # Confirm it was removed from the store entirely
        self.assertIsNone(get_session(session.session_id))

    def test_cleanup_expired_sessions(self):
        """Verifies batch cleanup removes expired sessions while keeping active ones."""
        # Create 2 sessions with 10s lifetime and 1 session with 3600s lifetime
        s1 = create_session("boomika", lifetime_seconds=10)
        s2 = create_session("alex", lifetime_seconds=10)
        s3 = create_session("boomika", lifetime_seconds=3600)

        test_now = s1.created_at + timedelta(seconds=20)
        purged = cleanup_expired_sessions(now=test_now)
        self.assertEqual(purged, 2)

        # s1 and s2 should be gone, s3 should remain
        self.assertIsNone(get_session(s1.session_id, now=test_now))
        self.assertIsNone(get_session(s2.session_id, now=test_now))
        self.assertIsNotNone(get_session(s3.session_id, now=test_now))

    # --------------------------------------------------------------------------
    # 6. Central Cookie Constant & Email Support
    # --------------------------------------------------------------------------
    def test_session_cookie_name_constant(self):
        """Verifies central session cookie name is defined as expected."""
        self.assertEqual(SESSION_COOKIE_NAME, "session_id")

    def test_authenticate_valid_credentials_email_boomika(self):
        """Verifies authenticating boomika by email returns correct agent identity."""
        identity = authenticate_user("boomika@support.industrial.ai", "Industrial@2026")
        self.assertIsNotNone(identity)
        self.assertEqual(identity["username"], "boomika")
        self.assertEqual(identity["agent_name"], "Boomika")
        self.assertEqual(set(identity.keys()), {"username", "agent_name"})

    def test_authenticate_valid_credentials_email_alex(self):
        """Verifies authenticating alex by email returns correct agent identity."""
        identity = authenticate_user("alex@support.industrial.ai", "Machinery#400")
        self.assertIsNotNone(identity)
        self.assertEqual(identity["username"], "alex")
        self.assertEqual(identity["agent_name"], "Alex Chen")
        self.assertEqual(set(identity.keys()), {"username", "agent_name"})

    def test_create_session_stores_email(self):
        """Verifies session stores the agent email."""
        session = create_session("boomika@support.industrial.ai")
        self.assertEqual(session.username, "boomika")
        self.assertEqual(session.agent_name, "Boomika")
        self.assertEqual(session.email, "boomika@support.industrial.ai")

    # --------------------------------------------------------------------------
    # 7. Agent Registration (register_user)
    # --------------------------------------------------------------------------
    def test_register_user_success(self):
        """Verifies registering a new agent stores salted hash and allows authentication."""
        try:
            reg = register_user("Sarah Connor", "sarah@support.industrial.ai", "Cyberdyne#2026")
            self.assertEqual(reg["agent_name"], "Sarah Connor")
            self.assertEqual(reg["email"], "sarah@support.industrial.ai")

            # Must authenticate successfully
            auth_res = authenticate_user("sarah@support.industrial.ai", "Cyberdyne#2026")
            self.assertIsNotNone(auth_res)
            self.assertEqual(auth_res["agent_name"], "Sarah Connor")

            # Must never store plaintext password
            user_rec = USER_REGISTRY[reg["username"]]
            self.assertNotEqual(user_rec["password_hash"], "Cyberdyne#2026")
            self.assertNotIn("Cyberdyne#2026", str(user_rec))
            self.assertIsInstance(user_rec["salt"], bytes)
            self.assertGreater(len(user_rec["salt"]), 0)
        finally:
            reset_demo_registry()

    def test_register_user_missing_fields_raises(self):
        """Verifies register_user rejects missing or empty fields."""
        with self.assertRaises(ValueError):
            register_user("", "agent@test.ai", "ValidPassword123")
        with self.assertRaises(ValueError):
            register_user("Agent Name", "", "ValidPassword123")
        with self.assertRaises(ValueError):
            register_user("Agent Name", "agent@test.ai", "")

    def test_register_user_invalid_email_format_raises(self):
        """Verifies register_user rejects invalid email strings."""
        invalid_emails = ["notanemail", "missing@domain", "@nodomain.com", "spaces in@email.com"]
        for bad_email in invalid_emails:
            with self.assertRaises(ValueError):
                register_user("Agent Name", bad_email, "ValidPassword123")

    def test_register_user_password_shorter_than_10_raises(self):
        """Verifies register_user enforces minimum password length of 10 chars."""
        with self.assertRaises(ValueError):
            register_user("Agent Name", "short@test.ai", "Short9ch")  # 8 chars
        with self.assertRaises(ValueError):
            register_user("Agent Name", "short@test.ai", "123456789")  # 9 chars

    def test_register_user_duplicate_email_case_insensitive_raises(self):
        """Verifies duplicate registration is rejected even with different case or whitespace."""
        try:
            register_user("Agent One", "duplicate@test.ai", "ValidPassword123")
            # Same email lowercase
            with self.assertRaises(ValueError):
                register_user("Agent Two", "duplicate@test.ai", "ValidPassword123")
            # Same email uppercase
            with self.assertRaises(ValueError):
                register_user("Agent Three", "DUPLICATE@TEST.AI", "ValidPassword123")
            # Same email with leading/trailing whitespace
            with self.assertRaises(ValueError):
                register_user("Agent Four", "  duplicate@test.ai  ", "ValidPassword123")
            # Duplicate demo email
            with self.assertRaises(ValueError):
                register_user("Fake Boomika", "Boomika@support.industrial.ai", "ValidPassword123")
        finally:
            reset_demo_registry()

    def test_demo_accounts_preserved_after_registration(self):
        """Verifies existing demo accounts boomika and alex remain intact after new registrations."""
        try:
            register_user("Temp Agent", "temp@test.ai", "ValidPassword123")
            # Boomika still authenticates
            b_auth = authenticate_user("boomika@support.industrial.ai", "Industrial@2026")
            self.assertIsNotNone(b_auth)
            self.assertEqual(b_auth["agent_name"], "Boomika")
            # Alex still authenticates
            a_auth = authenticate_user("alex@support.industrial.ai", "Machinery#400")
            self.assertIsNotNone(a_auth)
            self.assertEqual(a_auth["agent_name"], "Alex Chen")
        finally:
            reset_demo_registry()


if __name__ == "__main__":
    unittest.main()
