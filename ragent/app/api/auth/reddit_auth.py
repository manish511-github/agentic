from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from app.auth.services.reddit_auth import oauth, get_reddit_user_tokens, refresh_reddit_token
from app.models import OAuthAccount, UserModel, OAuthState
from app.database import get_sync_db
from app.auth.security import get_current_user
from sqlalchemy import select
from datetime import datetime
from app.settings import get_settings
import logging
import uuid

settings = get_settings()
FRONTEND_URL = settings.FRONTEND_HOST

reddit_auth_router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger(__name__)

@reddit_auth_router.get("/reddit")
async def reddit_login(request: Request, current_user: UserModel = Depends(get_current_user), db=Depends(get_sync_db)):
    state = str(uuid.uuid4())
    oauth_state = OAuthState(state=state, user_id=current_user.id)
    db.add(oauth_state)
    db.commit()
    return await oauth.reddit.authorize_redirect(request, settings.REDDIT_REDIRECT_URI, state=state)

@reddit_auth_router.get("/callback/reddit")
async def reddit_callback(
    request: Request,
    db=Depends(get_sync_db),
):
    state = request.query_params.get("state")
    oauth_state = db.query(OAuthState).filter_by(state=state).first()
    if not oauth_state:
        raise HTTPException(status_code=401, detail="Invalid or expired state")
    user_id = oauth_state.user_id
    # Clean up state
    db.delete(oauth_state)
    db.commit()
    # Get user
    current_user = db.query(UserModel).filter_by(id=user_id).first()
    if not current_user:
        raise HTTPException(status_code=401, detail="User not found")
    tokens, reddit_user_id = await get_reddit_user_tokens(request)
    if not tokens:
        raise HTTPException(status_code=400, detail="Reddit authentication failed.")
    logger.info(f"Reddit authentication successful for user {current_user.id}")

    # Store or update OAuthAccount for this user
    result = db.execute(
        select(OAuthAccount).where(
            OAuthAccount.user_id == current_user.id,
            OAuthAccount.provider == 'reddit'
        )
    )
    oauth_account = result.scalars().first()
    if not oauth_account:
        oauth_account = OAuthAccount(
            user_id=current_user.id,
            provider='reddit',
            provider_user_id=reddit_user_id,
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            token_expiry=tokens['expires_at'],
            scope=tokens.get('scope', []),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(oauth_account)
    else:
        oauth_account.access_token = tokens['access_token']
        oauth_account.refresh_token = tokens['refresh_token']
        oauth_account.token_expiry = tokens['expires_at']
        oauth_account.scope = tokens.get('scope', [])
        oauth_account.updated_at = datetime.utcnow()
        db.add(oauth_account)
    db.commit()
    db.refresh(oauth_account)
   
    logger.info(f"Reddit account connected for user {current_user.id}")
    # Redirect to frontend (e.g., profile/settings page)
    return RedirectResponse(url=f"{FRONTEND_URL}/profile?reddit_connected=1")