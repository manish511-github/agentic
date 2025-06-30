
from fastapi import APIRouter, Depends, HTTPException
from app.models import OAuthAccount, UserModel
from app.database import get_sync_db
from app.auth.security import get_current_user
from sqlalchemy import select
from app.auth.services.reddit_auth import refresh_reddit_token
from app.settings import get_settings
import requests
from datetime import datetime

settings = get_settings()
REDDIT_CLIENT_ID = settings.REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET = settings.REDDIT_CLIENT_SECRET

reddit_post_router = APIRouter(prefix="/reddit", tags=["reddit"])

@reddit_post_router.post("/post")
async def post_to_reddit(
    subreddit: str,
    title: str,
    text: str,
    current_user: UserModel = Depends(get_current_user),
    db=Depends(get_sync_db)
):
    # 1. Get user's Reddit OAuthAccount
    result = db.execute(
        select(OAuthAccount).where(
            OAuthAccount.user_id == current_user.id,
            OAuthAccount.provider == 'reddit'
        )
    )
    reddit_account = result.scalars().first()
    if not reddit_account or not reddit_account.access_token:
        raise HTTPException(status_code=400, detail="Reddit account not connected.")

    # 2. Refresh token if expired
    if reddit_account.token_expiry and reddit_account.token_expiry < datetime.utcnow():
        refreshed = refresh_reddit_token(reddit_account, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        if not refreshed:
            raise HTTPException(status_code=400, detail="Could not refresh Reddit token.")
        db.add(reddit_account)
        db.commit()
        db.refresh(reddit_account)

    # 3. Post to Reddit API
    headers = {
        "Authorization": f"bearer {reddit_account.access_token}",
        "User-Agent": "YourAppName/0.1 by YourRedditUsername"
    }
    data = {
        "sr": subreddit,
        "title": title,
        "kind": "self",
        "text": text
    }
    response = requests.post("https://oauth.reddit.com/api/submit", headers=headers, data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Reddit API error: {response.text}")

    return {"message": "Post submitted!", "reddit_response": response.json()}