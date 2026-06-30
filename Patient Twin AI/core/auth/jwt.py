"""JWT verification → Principal (docs/07 §1).

Verification only — these services never *issue* tokens; an external auth
issuer signs them. Configurable so the same code runs with an asymmetric public
key (RS256/ES256, prod) or a shared secret (HS256, local dev). No key material
is hard-coded; it comes from the environment (`JWT_PUBLIC_KEY` / `JWT_SECRET`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import jwt

from .errors import AuthError
from .principal import Principal, Role


def _normalize_pem(value: str) -> str:
    # Allow single-line PEMs passed via env (literal "\n" → real newlines).
    return value.replace("\\n", "\n").strip()


def _parse_roles(raw: Any) -> frozenset[Role]:
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    roles: set[Role] = set()
    for item in raw:
        try:
            roles.add(Role(str(item).lower()))
        except ValueError:
            continue  # unknown roles are ignored, never elevate
    return frozenset(roles)


def _parse_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    raw = claims.get("scope", claims.get("scopes"))
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset(raw.split())
    return frozenset(str(s) for s in raw)


def _principal_from_claims(claims: Mapping[str, Any]) -> Principal:
    subject = claims.get("sub")
    if not subject:
        raise AuthError("token missing required 'sub' claim")
    patient_id: UUID | None = None
    raw_pid = claims.get("patient_id")
    if raw_pid:
        try:
            patient_id = UUID(str(raw_pid))
        except ValueError as exc:
            raise AuthError("token 'patient_id' claim is not a valid UUID") from exc
    return Principal(
        subject=str(subject),
        roles=_parse_roles(claims.get("roles")),
        token_scopes=_parse_scopes(claims),
        patient_id=patient_id,
    )


class JWTVerifier:
    def __init__(
        self,
        *,
        key: str,
        algorithms: tuple[str, ...],
        audience: str | None = None,
        issuer: str | None = None,
        leeway: int = 0,
    ) -> None:
        self._key = key
        self._algorithms = tuple(algorithms)
        self._audience = audience
        self._issuer = issuer
        self._leeway = leeway

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> JWTVerifier:
        env = env if env is not None else os.environ
        public_key = (env.get("JWT_PUBLIC_KEY") or "").strip()
        secret = (env.get("JWT_SECRET") or "").strip()
        audience = env.get("JWT_AUDIENCE") or None
        issuer = env.get("JWT_ISSUER") or None
        if public_key:
            return cls(
                key=_normalize_pem(public_key),
                algorithms=("RS256", "ES256"),
                audience=audience,
                issuer=issuer,
            )
        if secret:
            return cls(key=secret, algorithms=("HS256",), audience=audience, issuer=issuer)
        raise AuthError(
            "no JWT verification key configured — set JWT_PUBLIC_KEY (prod) or JWT_SECRET (dev)"
        )

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={
                    "require": ["exp", "sub"],
                    "verify_aud": self._audience is not None,
                },
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        return _principal_from_claims(claims)
