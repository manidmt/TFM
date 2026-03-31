'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Password hashing and verification using stdlib scrypt (rpi5.md §14.1).

Uses Python's built-in ``hashlib.scrypt`` (OpenSSL-backed, available since
Python 3.6).  No third-party dependency required.

Hash format (colon-separated)
------------------------------
    scrypt:<base64-salt>:<base64-derived-key>

Example
-------
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)

Cost parameters
---------------
Module-level ``SCRYPT_N / SCRYPT_R / SCRYPT_P`` are the scrypt cost parameters.
Tests may lower ``SCRYPT_N`` to ``2`` for speed without affecting correctness::

    import quant_risk.prod.auth.password as pw_module
    pw_module.SCRYPT_N = 2

Security note
-------------
The 16-byte random salt, n=16384, r=8, p=1 parameters match the OWASP
scrypt recommendation for interactive logins (2024).  The derived key is
32 bytes (256 bits), constant-time compared via ``hmac.compare_digest``.
'''

from __future__ import annotations

import base64
import hashlib
import hmac
import os

# ---------------------------------------------------------------------------
# Cost parameters — lower SCRYPT_N in tests for speed
# ---------------------------------------------------------------------------

SCRYPT_N: int = 16384  # CPU/memory cost (must be power of 2)
SCRYPT_R: int = 8       # block size
SCRYPT_P: int = 1       # parallelisation
_DKLEN: int = 32        # derived key length in bytes
_SALT_LEN: int = 16     # random salt length in bytes

MIN_PASSWORD_LEN: int = 8

_PREFIX = "scrypt"


def hash_password(plain: str) -> str:
    """Hash *plain* with scrypt and return a storable string.

    Parameters
    ----------
    plain:
        Plaintext password.  Must be at least ``MIN_PASSWORD_LEN`` characters.

    Returns
    -------
    str
        ``"scrypt:<base64-salt>:<base64-derived-key>"``

    Raises
    ------
    ValueError
        If *plain* is shorter than ``MIN_PASSWORD_LEN``.
    """
    if len(plain) < MIN_PASSWORD_LEN:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LEN} characters, "
            f"got {len(plain)}."
        )
    salt = os.urandom(_SALT_LEN)
    dk = hashlib.scrypt(
        plain.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=_DKLEN,
    )
    b64_salt = base64.b64encode(salt).decode("ascii")
    b64_dk = base64.b64encode(dk).decode("ascii")
    return f"{_PREFIX}:{b64_salt}:{b64_dk}"


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*.

    Always returns False (never raises) on malformed hash strings.
    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """
    try:
        prefix, b64_salt, b64_dk = hashed.split(":")
        if prefix != _PREFIX:
            return False
        salt = base64.b64decode(b64_salt)
        expected = base64.b64decode(b64_dk)
    except Exception:
        return False

    actual = hashlib.scrypt(
        plain.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=_DKLEN,
    )
    return hmac.compare_digest(actual, expected)
