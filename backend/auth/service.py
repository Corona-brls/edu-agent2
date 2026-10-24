"""JWT token and local authentication service."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.auth.password import hash_password, verify_password
from backend.core.config import get_settings
from backend.storage.logger import LearningLogger


class AuthError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if secret:
        return secret
    raise AuthError("JWT_SECRET_MISSING", "JWT_SECRET_KEY 未配置。")


def authenticate_user(storage: LearningLogger, username: str, password: str) -> dict[str, Any]:
    user = storage.user_by_username(username)
    if not user or not bool(user["enabled"]):
        raise AuthError("INVALID_CREDENTIALS", "用户名或密码错误。")
    if not verify_password(password, user["password_hash"]):
        raise AuthError("INVALID_CREDENTIALS", "用户名或密码错误。")
    storage.mark_user_login(int(user["id"]))
    return storage.user(int(user["id"])) or user


def change_password(storage: LearningLogger, user_id: int, current_password: str, new_password: str) -> dict[str, Any]:
    user = storage.user(user_id)
    if not user or not bool(user["enabled"]):
        raise AuthError("USER_DISABLED", "用户不存在或已禁用。")
    if not verify_password(current_password, user["password_hash"]):
        raise AuthError("INVALID_CREDENTIALS", "当前密码不正确。")
    if verify_password(new_password, user["password_hash"]):
        raise AuthError("PASSWORD_UNCHANGED", "新密码不能与当前密码相同。")
    storage.update_user_password(user_id, hash_password(new_password), must_change_password=False)
    updated = storage.user(user_id)
    if updated is None:
        raise AuthError("USER_DISABLED", "用户不存在或已禁用。")
    return updated


def create_access_token(user: dict[str, Any], expires_delta: timedelta | None = None) -> tuple[str, int]:
    settings = get_settings()
    minutes = settings.access_token_expire_minutes
    expires = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=minutes))
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "exp": expires,
        "jti": secrets.token_hex(8),
        "token_version": int(user.get("token_version", 0)),
    }
    token = jwt.encode(payload, jwt_secret(), algorithm=settings.jwt_algorithm)
    return token, int((expires - datetime.now(timezone.utc)).total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=[get_settings().jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("TOKEN_EXPIRED", "访问令牌已过期。") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("INVALID_TOKEN", "访问令牌无效。") from exc
    if "sub" not in payload:
        raise AuthError("INVALID_TOKEN", "访问令牌缺少用户标识。")
    return payload


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
        "enabled": bool(user["enabled"]),
        "must_change_password": bool(user.get("must_change_password", 0)),
    }
