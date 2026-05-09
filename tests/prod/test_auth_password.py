'''
Tests for quant_risk.prod.auth.password

Covers:
- hash_password produces a valid scrypt: prefix
- verify_password returns True for correct password
- verify_password returns False for wrong password
- verify_password is constant-time (does not raise)
- minimum length enforcement
- different salts produce different hashes for the same password
- malformed hash strings handled gracefully
- module-level SCRYPT_N is lowered for test speed
'''

import pytest

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.password import hash_password, verify_password

# Speed up tests: lower scrypt cost
pw_mod.SCRYPT_N = 2


class TestHashPassword:
    def test_returns_scrypt_prefix(self):
        h = hash_password("correct horse")
        assert h.startswith("scrypt:")

    def test_format_has_three_parts(self):
        h = hash_password("correct horse")
        parts = h.split(":")
        assert len(parts) == 3

    def test_same_password_different_hashes(self):
        """Each call uses a fresh random salt."""
        h1 = hash_password("correct horse")
        h2 = hash_password("correct horse")
        assert h1 != h2

    def test_minimum_length_exact(self):
        # 8 chars — must succeed
        h = hash_password("abcdefgh")
        assert h.startswith("scrypt:")

    def test_minimum_length_too_short_raises(self):
        with pytest.raises(ValueError, match="at least"):
            hash_password("short")

    def test_empty_password_raises(self):
        with pytest.raises(ValueError):
            hash_password("")

    def test_unicode_password(self):
        h = hash_password("contraseña123")
        assert verify_password("contraseña123", h)

    def test_long_password(self):
        h = hash_password("x" * 100)
        assert verify_password("x" * 100, h)


class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        h = hash_password("correct horse")
        assert verify_password("correct horse", h) is True

    def test_wrong_password_returns_false(self):
        h = hash_password("correct horse")
        assert verify_password("wrong password", h) is False

    def test_empty_plain_returns_false(self):
        h = hash_password("correct horse")
        assert verify_password("", h) is False

    def test_malformed_hash_no_colon_returns_false(self):
        assert verify_password("password", "notahash") is False

    def test_malformed_hash_wrong_prefix_returns_false(self):
        assert verify_password("password", "md5:abc:def") is False

    def test_malformed_hash_invalid_base64_returns_false(self):
        assert verify_password("password", "scrypt:!!!:???") is False

    def test_empty_hash_returns_false(self):
        assert verify_password("password", "") is False

    def test_hash_of_different_password_returns_false(self):
        h = hash_password("password_one")
        assert verify_password("password_two", h) is False

    def test_case_sensitive(self):
        h = hash_password("Password123")
        assert verify_password("password123", h) is False
        assert verify_password("Password123", h) is True
