"""
Authentication and authorization for the Apex Harness API.

Supports:
  - API key authentication (SHA-256 hashed keys)
  - OAuth2 with PKCE (client_credentials, authorization_code, refresh_token)
  - mTLS (client certificate verification)
  - RBAC (role-based access control) with roles: admin, operator, viewer, service
  - ABAC (attribute-based access control) via scopes and tenant isolation
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from models import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyInfo,
    AuthScheme,
    Role,
    TokenRequest,
    TokenResponse,
    UserContext,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY_HEADER = "X-API-Key"
AUTHORIZATION_HEADER = "Authorization"
MTLS_CLIENT_CERT_HEADER = "X-Client-Cert-Verified"  # Set by reverse proxy / TLS terminator

# In-memory stores (replace with persistent storage in production)
_api_keys: dict[str, dict] = {}  # key_id -> key record
_oauth_clients: dict[str, dict] = {}  # client_id -> client record
_access_tokens: dict[str, dict] = {}  # token -> token record
_refresh_tokens: dict[str, dict] = {}  # refresh_token -> token record

# Role hierarchy: higher index = more permissions
_ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.SERVICE: 2,
    Role.ADMIN: 3,
}

# Default scopes per role
_ROLE_SCOPES: dict[Role, list[str]] = {
    Role.VIEWER: ["read"],
    Role.OPERATOR: ["read", "write", "execute"],
    Role.SERVICE: ["read", "write", "execute", "route"],
    Role.ADMIN: ["read", "write", "execute", "route", "admin", "governance"],
}


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------


def _hash_api_key(raw_key: str) -> str:
    """Hash an API key using SHA-256 with a pepper."""
    pepper = b"apex-harness-v1"  # In production, load from env / secrets manager
    return hashlib.sha256(raw_key.encode() + pepper).hexdigest()


def _generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return f"apex_{secrets.token_urlsafe(48)}"


def create_api_key(request: ApiKeyCreateRequest) -> ApiKeyCreateResponse:
    """Create a new API key. Returns the raw key (shown only once)."""
    raw_key = _generate_api_key()
    key_id = str(uuid.uuid4())
    hashed = _hash_api_key(raw_key)

    now = datetime.utcnow()
    expires_at = now + timedelta(days=request.expires_in_days) if request.expires_in_days else None

    record = {
        "key_id": key_id,
        "hashed_key": hashed,
        "name": request.name,
        "role": request.role,
        "tenant_id": request.tenant_id,
        "scopes": request.scopes or _ROLE_SCOPES.get(request.role, ["read"]),
        "created_at": now,
        "expires_at": expires_at,
        "last_used_at": None,
        "is_active": True,
    }
    _api_keys[key_id] = record

    return ApiKeyCreateResponse(
        key_id=key_id,
        api_key=raw_key,
        name=request.name,
        role=request.role,
        tenant_id=request.tenant_id,
        created_at=now,
        expires_at=expires_at,
    )


def revoke_api_key(key_id: str) -> bool:
    """Revoke an API key by ID. Returns True if found and revoked."""
    if key_id in _api_keys:
        _api_keys[key_id]["is_active"] = False
        return True
    return False


def list_api_keys(tenant_id: Optional[str] = None) -> list[ApiKeyInfo]:
    """List API keys, optionally filtered by tenant."""
    results = []
    for rec in _api_keys.values():
        if tenant_id and rec["tenant_id"] != tenant_id:
            continue
        results.append(
            ApiKeyInfo(
                key_id=rec["key_id"],
                name=rec["name"],
                role=rec["role"],
                tenant_id=rec["tenant_id"],
                created_at=rec["created_at"],
                expires_at=rec["expires_at"],
                last_used_at=rec["last_used_at"],
                is_active=rec["is_active"],
            )
        )
    return results


def _validate_api_key(raw_key: str) -> Optional[dict]:
    """Validate a raw API key. Returns the key record or None."""
    hashed = _hash_api_key(raw_key)
    for rec in _api_keys.values():
        if hmac.compare_digest(rec["hashed_key"], hashed):
            if not rec["is_active"]:
                return None
            if rec["expires_at"] and datetime.utcnow() > rec["expires_at"]:
                return None
            rec["last_used_at"] = datetime.utcnow()
            return rec
    return None


# ---------------------------------------------------------------------------
# OAuth2 + PKCE
# ---------------------------------------------------------------------------


def register_oauth_client(
    client_id: str,
    client_secret: str,
    redirect_uris: list[str],
    allowed_scopes: list[str],
) -> None:
    """Register an OAuth2 client."""
    _oauth_clients[client_id] = {
        "client_secret": client_secret,
        "redirect_uris": redirect_uris,
        "allowed_scopes": allowed_scopes,
    }


def _verify_pkce(code_verifier: str, code_challenge: str, method: str = "S256") -> bool:
    """Verify a PKCE code_verifier against a code_challenge."""
    if method == "plain":
        return hmac.compare_digest(code_verifier, code_challenge)
    # S256
    computed = hashlib.sha256(code_verifier.encode()).digest()
    import base64

    computed_b64 = base64.urlsafe_b64encode(computed).rstrip(b"=").decode()
    return hmac.compare_digest(computed_b64, code_challenge)


def issue_token(
    client_id: str,
    scopes: list[str],
    tenant_id: str = "default",
    role: Role = Role.SERVICE,
    ttl_seconds: int = 3600,
) -> TokenResponse:
    """Issue an OAuth2 access token."""
    token = secrets.token_urlsafe(64)
    refresh = secrets.token_urlsafe(64)
    expires_in = ttl_seconds

    _access_tokens[token] = {
        "client_id": client_id,
        "scopes": scopes,
        "tenant_id": tenant_id,
        "role": role,
        "expires_at": time.time() + expires_in,
    }
    _refresh_tokens[refresh] = {
        "access_token": token,
        "client_id": client_id,
    }

    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=expires_in,
        refresh_token=refresh,
        scope=" ".join(scopes),
    )


def handle_token_request(request: TokenRequest) -> TokenResponse:
    """Handle an OAuth2 token request."""
    client = _oauth_clients.get(request.client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client_id",
        )

    if request.grant_type == "client_credentials":
        if not request.client_secret or not hmac.compare_digest(
            request.client_secret, client["client_secret"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client_secret",
            )
        scopes = request.scope.split() if request.scope else client["allowed_scopes"]
        return issue_token(request.client_id, scopes)

    elif request.grant_type == "authorization_code":
        if not request.code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="code is required for authorization_code grant",
            )
        # In production: validate code against stored authorization codes
        scopes = request.scope.split() if request.scope else client["allowed_scopes"]
        return issue_token(request.client_id, scopes)

    elif request.grant_type == "refresh_token":
        if not request.refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="refresh_token is required",
            )
        stored = _refresh_tokens.get(request.refresh_token)
        if not stored:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh_token",
            )
        old_token_data = _access_tokens.pop(stored["access_token"], None)
        scopes = old_token_data["scopes"] if old_token_data else ["read"]
        return issue_token(stored["client_id"], scopes)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported grant_type: {request.grant_type}",
    )


def _validate_access_token(token: str) -> Optional[dict]:
    """Validate an OAuth2 access token. Returns token data or None."""
    data = _access_tokens.get(token)
    if not data:
        return None
    if time.time() > data["expires_at"]:
        _access_tokens.pop(token, None)
        return None
    return data


# ---------------------------------------------------------------------------
# mTLS
# ---------------------------------------------------------------------------


def verify_mtls(request: Request) -> Optional[UserContext]:
    """
    Verify mTLS client certificate.

    In production, the reverse proxy (nginx, Envoy, etc.) terminates TLS and
    forwards the client certificate verification result via headers.
    This function checks for those headers.
    """
    verified = request.headers.get(MTLS_CLIENT_CERT_HEADER, "").lower()
    if verified not in ("true", "1", "yes"):
        return None

    # Extract client identity from headers set by the TLS terminator
    client_id = request.headers.get("X-Client-Cert-Subject-CN", "mtls-client")
    tenant_id = request.headers.get("X-Client-Cert-Tenant", "default")

    return UserContext(
        principal_id=client_id,
        role=Role.SERVICE,
        tenant_id=tenant_id,
        auth_scheme=AuthScheme.MTLS,
        scopes=_ROLE_SCOPES[Role.SERVICE],
    )


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Security(_api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> UserContext:
    """
    Authenticate the request using API key, OAuth2 Bearer token, or mTLS.

    Priority: API key > OAuth2 Bearer > mTLS
    """
    # 1. API key
    if api_key:
        record = _validate_api_key(api_key)
        if record:
            return UserContext(
                principal_id=record["key_id"],
                role=record["role"],
                tenant_id=record["tenant_id"],
                auth_scheme=AuthScheme.API_KEY,
                scopes=record["scopes"],
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    # 2. OAuth2 Bearer
    if credentials and credentials.scheme == "Bearer":
        token_data = _validate_access_token(credentials.credentials)
        if token_data:
            return UserContext(
                principal_id=token_data["client_id"],
                role=token_data["role"],
                tenant_id=token_data["tenant_id"],
                auth_scheme=AuthScheme.OAUTH2,
                scopes=token_data["scopes"],
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )

    # 3. mTLS
    mtls_user = verify_mtls(request)
    if mtls_user:
        return mtls_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide an API key, OAuth2 Bearer token, or mTLS client certificate.",
    )


def require_role(min_role: Role):
    """
    Dependency factory: require at least the given role.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: UserContext = Depends(require_role(Role.ADMIN))):
            ...
    """

    def _check(user: UserContext = Depends(get_current_user)) -> UserContext:
        if _ROLE_HIERARCHY.get(user.role, -1) < _ROLE_HIERARCHY.get(min_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {min_role.value}",
            )
        return user

    return _check


def require_scope(required_scope: str):
    """
    Dependency factory: require a specific scope (ABAC).

    Usage:
        @app.post("/execute")
        async def execute(user: UserContext = Depends(require_scope("execute"))):
            ...
    """

    def _check(user: UserContext = Depends(get_current_user)) -> UserContext:
        if required_scope not in user.scopes and user.role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {required_scope}",
            )
        return user

    return _check


def require_tenant_access(target_tenant_id: str):
    """
    Dependency factory: ensure the authenticated user belongs to the target tenant.
    Admins can access all tenants.
    """

    def _check(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role == Role.ADMIN:
            return user
        if user.tenant_id != target_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied for tenant: {target_tenant_id}",
            )
        return user

    return _check
