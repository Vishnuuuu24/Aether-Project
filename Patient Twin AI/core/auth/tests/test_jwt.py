"""JWT verification → Principal. Asymmetric (RS256) and symmetric (HS256) paths,
plus the rejection cases (expired, missing sub, bad signature, no key configured).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from core.auth import AuthError, JWTVerifier, Role


def _rsa_pem() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    pub = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


def _exp(hours: int) -> datetime:
    return datetime.now(UTC) + timedelta(hours=hours)


def test_rs256_roundtrip_to_principal() -> None:
    priv, pub = _rsa_pem()
    pid = uuid4()
    token = jwt.encode(
        {
            "sub": "patient-abc",
            "roles": ["patient"],
            "scope": "vitals copilot",
            "patient_id": str(pid),
            "exp": _exp(1),
        },
        priv,
        algorithm="RS256",
    )
    principal = JWTVerifier(key=pub, algorithms=("RS256",)).verify(token)
    assert principal.subject == "patient-abc"
    assert principal.roles == frozenset({Role.PATIENT})
    assert principal.token_scopes == {"vitals", "copilot"}
    assert principal.patient_id == pid


def test_unknown_roles_are_ignored_never_elevate() -> None:
    priv, pub = _rsa_pem()
    token = jwt.encode(
        {"sub": "x", "roles": ["patient", "superuser", "root"], "exp": _exp(1)},
        priv,
        algorithm="RS256",
    )
    principal = JWTVerifier(key=pub, algorithms=("RS256",)).verify(token)
    assert principal.roles == frozenset({Role.PATIENT})


def test_expired_token_rejected() -> None:
    priv, pub = _rsa_pem()
    token = jwt.encode({"sub": "x", "exp": _exp(-1)}, priv, algorithm="RS256")
    with pytest.raises(AuthError):
        JWTVerifier(key=pub, algorithms=("RS256",)).verify(token)


def test_missing_sub_rejected() -> None:
    priv, pub = _rsa_pem()
    token = jwt.encode({"exp": _exp(1)}, priv, algorithm="RS256")
    with pytest.raises(AuthError):
        JWTVerifier(key=pub, algorithms=("RS256",)).verify(token)


def test_bad_signature_rejected() -> None:
    priv, _ = _rsa_pem()
    _, other_pub = _rsa_pem()
    token = jwt.encode({"sub": "x", "exp": _exp(1)}, priv, algorithm="RS256")
    with pytest.raises(AuthError):
        JWTVerifier(key=other_pub, algorithms=("RS256",)).verify(token)


def test_hs256_from_env() -> None:
    verifier = JWTVerifier.from_env({"JWT_SECRET": "dev-secret"})
    token = jwt.encode(
        {"sub": "svc", "roles": ["system"], "exp": _exp(1)}, "dev-secret", algorithm="HS256"
    )
    principal = verifier.verify(token)
    assert principal.is_system


def test_from_env_without_key_raises() -> None:
    with pytest.raises(AuthError):
        JWTVerifier.from_env({})
