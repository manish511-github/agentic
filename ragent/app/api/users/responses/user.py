from typing import Union, Optional
from datetime import datetime
from pydantic import EmailStr
from app.api.users.responses.base import BaseResponse


class UserResponse(BaseResponse):
    id: int
    name: str 
    email: EmailStr
    is_active: bool
    created_at: Optional[Union[datetime, str]] = None

