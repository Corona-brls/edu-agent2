"""FastAPI dependencies for local auth and role checks."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException

from backend.auth.service import AuthError, decode_access_token
from backend.storage.logger import LearningLogger


def get_optional_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail={"error_code": "INVALID_AUTH_HEADER", "message": "Authorization 必须使用 Bearer token。"})
    try:
        payload = decode_access_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail={"error_code": exc.error_code, "message": str(exc)}) from exc
    from backend.main import learning_logger

    user = learning_logger.user(int(payload["sub"]))
    if not user or not bool(user["enabled"]):
        raise HTTPException(status_code=401, detail={"error_code": "USER_DISABLED", "message": "用户不存在或已禁用。"})
    if int(payload.get("token_version", 0)) != int(user.get("token_version", 0)):
        raise HTTPException(status_code=401, detail={"error_code": "TOKEN_REVOKED", "message": "访问令牌已失效，请重新登录。"})
    return user


def get_current_user(user: Annotated[dict | None, Depends(get_optional_current_user)]) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail={"error_code": "AUTH_REQUIRED", "message": "请先登录。"})
    return user


def require_roles(*roles: str):
    allowed = {role.lower() for role in roles}

    def dependency(user: Annotated[dict, Depends(get_current_user)]) -> dict:
        if str(user["role"]).lower() not in allowed:
            raise HTTPException(status_code=403, detail={"error_code": "FORBIDDEN", "message": "当前角色无权访问该接口。"})
        return user

    return dependency


def can_access_user_resource(current_user: dict, target_user_id: str | int) -> bool:
    role = str(current_user["role"]).lower()
    if role == "admin":
        return True
    if role == "student":
        return str(current_user["id"]) == str(target_user_id) or current_user["username"] == str(target_user_id)
    if role == "teacher":
        from backend.main import learning_logger

        return learning_logger.teacher_can_access_user(int(current_user["id"]), target_user_id)
    return False
