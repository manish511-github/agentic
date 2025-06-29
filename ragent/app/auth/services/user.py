from fastapi import HTTPException
from app.models import UserModel, UserToken
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import joinedload

import logging

from app.auth.security import get_token_payload, hash_password, is_password_strong_enough, str_decode, str_encode, verify_password,load_user
from app.auth.schemas.users import UserResponse
from app.core.email.service.auth.auth_email_service import send_account_activation_confirmation_email, send_account_verification_email, send_password_reset_email, send_password_updated_email
from app.core.email.context.auth.auth_context import FORGOT_PASSWORD, USER_VERIFY_ACCOUNT
from app.auth.security import unique_string, generate_token
from app.settings import get_settings
settings = get_settings() 
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

async def get_login_token(data, session):
    # verify the email and password
    # Verify that user account is verified
    # Verify user account is active
    # generate access_token and refresh_token and ttl
    
    user = await load_user(data.username, session)
    if not user:
        raise HTTPException(status_code=400, detail="Email is not registered with us.")
    
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password.")
    
    if not user.verified_at:
        raise HTTPException(status_code=400, detail="Your account is not verified. Please check your email inbox to verify your account.")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Your account has been dactivated. Please contact support.")
        
    # Generate the JWT Token
    return _generate_tokens(user, session)

def _generate_tokens(user, session):
    """
    Generates access and refresh tokens for a user.

    This function generates a unique access token and refresh token for a user, along with their expiration times. The tokens are stored in the database and returned as part of the response.

    Args:
        user (UserModel): The user model instance.
        session (Session): The database session.

    Returns:
        dict: A dictionary containing the access token, refresh token, and their expiration times.
    """
    # Access token -> Used to access protected resources (like APIs).
    # Refresh token -> Used to get a new access token when the old one expires.

    #Create refresh key and acess key
    refresh_key = unique_string(100)
    access_key = unique_string(50)
    rt_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    
    #Store refresh key and access key in db
    user_token = UserToken()
    user_token.user_id = user.id
    user_token.refresh_key = refresh_key
    user_token.access_key = access_key
    user_token.expires_at = datetime.utcnow() + rt_expires
    session.add(user_token)
    session.commit()
    session.refresh(user_token)

    #Generate access token using jwt with at payload
    at_payload = {
        "sub": str_encode(str(user.id)),
        'a': access_key,
        'r': str_encode(str(user_token.id)),
        'n': str_encode(f"{user.username}")
    }

    at_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = generate_token(at_payload, settings.JWT_SECRET, settings.JWT_ALGORITHM, at_expires)

    #Generate refresh token 
    rt_payload = {"sub": str_encode(str(user.id)), "t": refresh_key, 'a': access_key}
    refresh_token = generate_token(rt_payload, settings.SECRET_KEY, settings.JWT_ALGORITHM, rt_expires)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": at_expires.seconds
    }


async def get_refresh_token(refresh_token, session):
    """
    Validates and processes a refresh token to generate new access and refresh tokens.

    Args:
        refresh_token (str): The refresh token to validate and use for token regeneration.
        session (Session): The database session to interact with the UserToken model.

    Returns:
        dict: A dictionary containing the new access token, refresh token, and their expiration times.
    """
    # Extract payload from the refresh token
    token_payload = get_token_payload(refresh_token, settings.SECRET_KEY, settings.JWT_ALGORITHM)
    if not token_payload:
        raise HTTPException(status_code=400, detail="Invalid Request.")
    
    # Extract necessary information from the payload
    refresh_key = token_payload.get('t')
    access_key = token_payload.get('a')
    user_id = str_decode(token_payload.get('sub'))
    
    # Query the database for a valid UserToken
    result = session.execute(
        select(UserToken)
        .options(joinedload(UserToken.user))
        .where(
            UserToken.refresh_key == refresh_key,
            UserToken.access_key == access_key,
            UserToken.user_id == user_id,
            UserToken.expires_at > datetime.utcnow()
        )
    )
    user_token = result.scalars().first()
    if not user_token:
        raise HTTPException(status_code=400, detail="Invalid Request.")
    
    # Update the UserToken's expiration time and save the changes
    user_token.expires_at = datetime.utcnow()
    session.add(user_token)
    session.commit()
    
    # Generate and return new tokens
    return _generate_tokens(user_token.user, session)

async def email_forgot_password_link(data, background_tasks, session):
    user = await load_user(data.email, session)
    if not user.verified_at:
        raise HTTPException(status_code=400, detail="Your account is not verified. Please check your email inbox to verify your account.")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Your account has been dactivated. Please contact support.")
    
    await send_password_reset_email(user, background_tasks)


async def reset_user_password(data, background_tasks, session):
    user = await load_user(data.email, session)
    
    if not user:
        logging.error("User not found during password reset")
        raise HTTPException(status_code=400, detail="Invalid request")
        
    if not user.verified_at:
        logging.error("User account is not verified during password reset")
        raise HTTPException(status_code=400, detail="Invalid request")
    
    if not user.is_active:
        logging.error("Inactive user account during password reset")
        raise HTTPException(status_code=400, detail="Invalid request")
    
    user_token = user.get_context_string(context=FORGOT_PASSWORD)
    try:
        token_valid = verify_password(user_token, data.token)
    except Exception as verify_exec:
        logging.exception("Token verification failed during password reset", exc_info=verify_exec)
        token_valid = False
    if not token_valid:
        logging.error("Invalid token provided during password reset")
        raise HTTPException(status_code=400, detail="Invalid window.")
    
    user.hashed_password = hash_password(data.password)
    logging.info("Password has been updated for user.")
    user.updated_at = datetime.now()
    session.add(user)
    session.commit()
    session.refresh(user)
    # Notify user that password has been updated
    await send_password_updated_email(user, background_tasks)

async def fetch_user_detail(pk, session):
    user = session.query(UserModel).filter(UserModel.id == pk).first()
    if user:
        return user
    raise HTTPException(status_code=400, detail="User does not exists.")