from fastapi import APIRouter, Request, Depends, HTTPException
from app.auth.services import google_auth
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_sync_db
import os
from app.settings import get_settings
from app.auth.services.google_auth import get_google_user, handle_google_oauth_callback
from app.auth.schemas.users import GoogleUser
from app.models import UserModel, OAuthAccount
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from datetime import datetime
settings = get_settings()

google_auth_router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = settings.GOOGLE_REDIRECT_URI
FRONTEND_URL = settings.FRONTEND_HOST

@google_auth_router.get("/google")
async def login_google(request: Request):
    return await google_auth.oauth.google.authorize_redirect(request, GOOGLE_REDIRECT_URI)

@google_auth_router.get("/callback/google")
async def auth_google(request: Request, db: AsyncSession = Depends(get_sync_db)):    
    user_info = await get_google_user(request)
    return await handle_google_oauth_callback(user_info, db)


    