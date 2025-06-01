from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    target_audience: Optional[str] = None
    website_url: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[datetime] = None
    budget: Optional[str] = None
    team: Optional[List[Dict]] = None
    tags: Optional[str] = None
    competitors: Optional[List[Dict]] = None
    keywords: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    owner_id: int
    created_at: datetime

    class Config:
        orm_mode = True
