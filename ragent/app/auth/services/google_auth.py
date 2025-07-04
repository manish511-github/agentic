from authlib.integrations.starlette_client import OAuth
from authlib.integrations.base_client import OAuthError
from starlette.config import Config
from authlib.oauth2.rfc6749 import OAuth2Token
from fastapi import HTTPException, status
from app.auth.schemas.users import GoogleUser
from app.models import UserModel, OAuthAccount
from sqlalchemy import select
from datetime import datetime
import os
import structlog
from app.settings import get_settings
from app.auth.services.user import _generate_tokens
from fastapi.responses import RedirectResponse
settings= get_settings()
logger = structlog.get_logger()


GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET

config_data = {'GOOGLE_CLIENT_ID': GOOGLE_CLIENT_ID, 'GOOGLE_CLIENT_SECRET': GOOGLE_CLIENT_SECRET}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)

oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

async def get_google_user(request):
    logger.info("Attempting to fetch user response from Google endpoint...")
    logger.info(request)
    try:
        user_response: OAuth2Token = await oauth.google.authorize_access_token(request)
        logger.info("Successfully fetched user response from Google endpoint.")
        logger.info(user_response)
    except OAuthError as e:
        logger.error("Failed to validate credentials with Google", error=str(e))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials with Google")
    
    user_info = user_response.get("userinfo")
    scope_str = user_response.get("scope", "")
    scope_list = scope_str.split() if scope_str else []
    user_info["scope"] = scope_list
    logger.info(f"User info fetched successfully.{user_info}")
    return user_info

async def handle_google_oauth_callback(user_info, db):
    google_user = GoogleUser(**user_info)

    # Check if user exists by email
    result = db.execute(select(UserModel).where(UserModel.email == google_user.email))
    user = result.scalars().first()

    if not user:
        # Create new user (no password for OAuth)
        user = UserModel(
            username=google_user.name,
            email=google_user.email,
            hashed_password='',  # No password for OAuth
            is_active=True,
            verified_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Check if OAuthAccount exists for this provider and user
    result = db.execute(
        select(OAuthAccount).where(
            OAuthAccount.provider == 'google',
            OAuthAccount.provider_user_id == str(google_user.sub),
            OAuthAccount.user_id == user.id
        )
    )
    oauth_account = result.scalars().first()

    if not oauth_account:
        oauth_account = OAuthAccount(
            user_id=user.id,
            provider='google',
            provider_user_id=str(google_user.sub),
            access_token=None,  # You can store tokens if needed
            refresh_token=None,
            token_expiry=None,
            scope=google_user.scope,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(oauth_account)
        db.commit()
        db.refresh(oauth_account)
    else:
        # Update scope if changed
        oauth_account.scope = google_user.scope
        oauth_account.updated_at = datetime.utcnow()
        db.add(oauth_account)
        db.commit()
        db.refresh(oauth_account)

    # Generate tokens using the same flow as user login
    tokens = _generate_tokens(user, db)
    # Build redirect URL with tokens as query parameters
    redirect_url = f"{settings.FRONTEND_HOST}/auth/callback?access_token={tokens['access_token']}&refresh_token={tokens['refresh_token']}&expires_in={tokens['expires_in']}"
    return RedirectResponse(url=redirect_url)
