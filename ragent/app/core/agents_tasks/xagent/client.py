from fastapi import HTTPException
from twikit import Client
from .x_agent_config import logger, TWITTER_USERNAME, TWITTER_EMAIL, TWITTER_PASSWORD

async def get_twitter_client(username, email, password, cookies_file):
    """Initialize and return a Twitter client with authentication for a specific account"""
    try:
        logger.info("Initializing Twitter client", username=username)
        client = Client('en-US')
        try:
            logger.info("Attempting to load existing cookies", cookies_file=cookies_file)
            await client.load_cookies(cookies_file)
            logger.info("Successfully loaded existing cookies")
        except Exception as e:
            logger.warning("No existing cookies found or invalid cookies", error=str(e))
            logger.info("Attempting new login", username=username, email=email)
            try:
                await client.login(
                    auth_info_1=username,
                    auth_info_2=email,
                    password=password,
                    cookies_file=cookies_file
                )
                logger.info("Successfully logged in and saved cookies")
            except Exception as login_error:
                logger.error("Login failed", error=str(login_error), username=username)
                raise HTTPException(
                    status_code=500,
                    detail=f"Twitter login failed: {str(login_error)}"
                )
        return client
    except Exception as e:
        logger.error("Failed to initialize Twitter client", error=str(e), stack_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Twitter client: {str(e)}"
        ) 