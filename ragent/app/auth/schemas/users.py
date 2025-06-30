from pydantic import BaseModel, EmailStr
from typing import Union, Optional, List
from datetime import datetime
from app.auth.schemas.base import BaseResponse

class RegisterUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseResponse):
    id: int
    name: str 
    email: EmailStr
    is_active: bool
    created_at: Optional[Union[datetime, str]] = None
    
class VerifyUserRequest(BaseModel):
    token: str
    email: EmailStr

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"

class EmailRequest(BaseModel):
    email: EmailStr
    
class ResetRequest(BaseModel):
    token: str
    email: EmailStr
    password: str

class GoogleUser(BaseModel):
    sub: int
    email: str
    name: str
    picture: str
    scope: List[str] = [] 