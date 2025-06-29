from datetime import datetime, timedelta
import logging
import jwt
import base64
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from app.settings import get_settings
import secrets
from app.models import UserToken
from sqlalchemy.orm import joinedload
from sqlalchemy import select
from app.database import get_sync_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException

settings = get_settings()


SPECIAL_CHARACTERS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>']
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hash_password):
    return pwd_context.verify(plain_password, hash_password)

def is_password_strong_enough(password: str) -> bool:
    if len(password) < 8:
        return False

    if not any(char.isupper() for char in password):
        return False

    if not any(char.islower() for char in password):
        return False

    if not any(char.isdigit() for char in password):
        return False

    if not any(char in SPECIAL_CHARACTERS for char in password):
        return False

    return True

async def load_user(email: str, db):
    from app.models import UserModel
    try:
        user = db.query(UserModel).filter(UserModel.email == email).first()
        logging.info(f"User found {user}")
    except Exception as user_exec:
        logging.info(f"User Not Found, Email: {email}")
        user = None
    return user

def str_encode(string: str) -> str:
    return base64.b85encode(string.encode('ascii')).decode('ascii')


def str_decode(string: str) -> str:
    return base64.b85decode(string.encode('ascii')).decode('ascii')


def unique_string(byte: int = 8) -> str:
    return secrets.token_urlsafe(byte)

def get_token_payload(token: str, secret: str, algo: str):
    try:
        payload = jwt.decode(token, secret, algorithms=algo)
    except Exception as jwt_exec:
        logging.debug(f"JWT Error: {str(jwt_exec)}")
        payload = None
    return payload

def generate_token(payload: dict, secret: str, algo: str, expiry: timedelta) -> str:
    """
    Generates a JWT token with the provided payload, secret, algorithm, and expiry duration.

    Args:
        payload (dict): The data you want to encode in the token (e.g., {"user_id": 123}).We are taking user_id,acess_token, user_name
        secret (str): A string key used to sign the token (keeps it secure).
        algo (str): The hashing algorithm (e.g., 'HS256').
        expiry (timedelta): A timedelta object that defines how long the token is valid (e.g., timedelta(minutes=15)).

    Returns:
        str: The generated JWT token.
    """
    expire = datetime.utcnow() + expiry
    payload.update({"exp": expire})
    return jwt.encode(payload, secret, algorithm=algo)

async def get_token_user(token: str, db):
    payload = get_token_payload(token, settings.JWT_SECRET, settings.JWT_ALGORITHM)
    if payload:
        user_token_id = str_decode(payload.get('r'))
        user_id = str_decode(payload.get('sub'))
        access_key = payload.get('a')
        stmt = select(UserToken).options(joinedload(UserToken.user)).where(
            UserToken.access_key == access_key,
            UserToken.id == user_token_id,
            UserToken.user_id == user_id,
            UserToken.expires_at > datetime.utcnow()
        )
        result = db.execute(stmt)
        user_token = result.scalars().first()
        if user_token:
            return user_token.user
    return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_sync_db)):
    user = await get_token_user(token=token, db=db)
    if user:
        return user
    raise HTTPException(status_code=401, detail="Not authorised.")