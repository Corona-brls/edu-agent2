from pydantic import BaseModel, Field


class UserPublic(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    enabled: bool = True
    must_change_password: bool = False


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic
    must_change_password: bool = False
    message: str = ""


class MeResponse(BaseModel):
    user: UserPublic


class LogoutResponse(BaseModel):
    logged_out: bool
    message: str


class ChangePasswordResponse(BaseModel):
    changed: bool
    must_change_password: bool = False
    message: str
    access_token: str | None = None
    token_type: str = "bearer"
    expires_in: int | None = None
    user: UserPublic | None = None


class TokenPayload(BaseModel):
    sub: str
    role: str
    username: str
    exp: int
