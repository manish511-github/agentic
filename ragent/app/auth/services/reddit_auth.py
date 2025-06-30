import logging
from authlib.integrations.starlette_client import OAuth
from app.settings import get_settings
from datetime import datetime, timedelta

settings = get_settings()
REDDIT_CLIENT_ID = settings.REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET = settings.REDDIT_CLIENT_SECRET
REDDIT_REDIRECT_URI = settings.REDDIT_REDIRECT_URI

oauth = OAuth()
oauth.register(
    name='reddit',
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    access_token_url='https://www.reddit.com/api/v1/access_token',
    authorize_url='https://www.reddit.com/api/v1/authorize',
    api_base_url='https://oauth.reddit.com',
    client_kwargs={
        'scope': 'identity submit',
        'token_endpoint_auth_method': 'client_secret_basic',
    },
)

async def get_reddit_user_tokens(request):
    """Retrieve Reddit user tokens and user information after authorization."""
    logger = logging.getLogger(__name__)
    try:
        # Authorize access token for Reddit
        try:

            token = await oauth.reddit.authorize_access_token(request)
        except Exception as e:
            logger.error(f"Failed to authorize access token: {e}")
            raise

        logger.info("Reddit access token authorized successfully.")
        # Fetch user information from Reddit API
        userinfo = await oauth.reddit.get('api/v1/me', token=token)
        logger.info("Reddit user informaton fetched successfully.")
        # Extract Reddit user ID from the user information
        reddit_user_id = str(userinfo.json()['id'])
        logger.info(f"Reddit user ID extracted: {reddit_user_id}")
        # Calculate the token expiration time
        expires_at = datetime.utcnow() + timedelta(seconds=token['expires_in'])
        # Prepare the token information to be returned
        token_info = {
            'access_token': token['access_token'],
            'refresh_token': token.get('refresh_token'),
            'expires_at': expires_at,
            'scope': token.get('scope', '').split()
        }
        logger.info("Token information prepared for return.")
        # Return the token information and the Reddit user ID
        return token_info, reddit_user_id
    except Exception as e:
        logger.error(f"Error during Reddit authentication: {e}")
        raise

def refresh_reddit_token(oauth_account, client_id, client_secret):
    import requests, requests.auth
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': oauth_account.refresh_token,
    }
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    response = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        data=data,
        auth=auth,
        headers={'User-Agent': 'YourAppName/0.1 by YourRedditUsername'}
    )
    if response.status_code == 200:
        token = response.json()
        oauth_account.access_token = token['access_token']
        oauth_account.token_expiry = datetime.utcnow() + timedelta(seconds=token['expires_in'])
        return True
    return False