from fastapi import HTTPException
from app.models import UserModel
from datetime import datetime, timedelta
from sqlalchemy import select
import logging

from app.api.users.security import hash_password, is_password_strong_enough, verify_password
from app.api.users.responses.user import UserResponse
from app.api.users.services.email_service import send_account_activation_confirmation_email, send_account_verification_email
from app.utils.email_context import USER_VERIFY_ACCOUNT

logger = logging.getLogger(__name__)

async def create_user_account(data, session, background_tasks):
    result = session.execute(select(UserModel).where(UserModel.email == data.email))
    user_exist = result.scalars().first()
    if user_exist:
        raise HTTPException(status_code=400, detail="Email is already exists.")

    if not is_password_strong_enough(data.password):
        raise HTTPException(status_code=400, detail="Please provide a strong password.")

    user = UserModel()
    user.username = data.name
    user.email = data.email
    user.hashed_password = hash_password(data.password)
    user.is_active = False
    user.updated_at = datetime.utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)

    # Account Verification Email
    await send_account_verification_email(user, background_tasks =background_tasks)
    return UserResponse(
        id=user.id,
        name=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at,
    )

async def activate_user_account(data, session, background_tasks):

    # Fetch user from DB
    result = session.execute(select(UserModel).where(UserModel.email == data.email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid link provided.")
    
    #Create the context from DB
    user_token = user.get_context_string(context=USER_VERIFY_ACCOUNT)

    # Compare the created context and recieved context
    try:
        token_valid = verify_password(user_token,data.token)
    except Exception as verify_exec:
        logger.exception(f"Verification failed: {verify_exec}")
        token_valid=False
    if not token_valid :
        raise HTTPException(status_code=400, detail="This link either expired or not valid.")
    
    # Update is_active status of user
    user.is_active=True
    user.updated_at = datetime.utcnow()
    user.verified_at = datetime.utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)

    #Activation cofirmation email
    await send_account_activation_confirmation_email(user, background_tasks)
    return user