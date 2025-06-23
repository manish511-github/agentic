import os
import logging
import sys
from logging.handlers import RotatingFileHandler
import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_EMAIL = os.getenv("TWITTER_EMAIL")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")

# Constants for rate limiting
MIN_WAIT_TIME = 60  # Increased to 60 seconds between requests
MAX_WAIT_TIME = 120  # Increased to 120 seconds between requests
MIN_TWEETS_PER_HASHTAG = 10

# Goal mapping for Twitter agent
GOAL_MAPPING = {
    "lead_generation": "grow web traffic",
    "brand_awareness": "increase brand awareness",
    "engagement": "engage potential customers",
    "support": "engage potential customers"
}

# Valid goals for validation
VALID_GOALS = ["increase brand awareness", "engage potential customers", "grow web traffic"]

def setup_logging():
    """Setup logging configuration"""
    # Standard logging configuration
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        level=logging.INFO,
        handlers=[
            RotatingFileHandler(
                'twitter_agent.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()

# Initialize logger
logger = setup_logging()

def load_twitter_accounts_from_env():
    """Load multiple Twitter account credentials from environment variables."""
    # accounts = []
    accounts = [
        {
            "username":"@ManishS1581951",
            "email": "singh.maneesh50@gmail.com",
            "password": "Maneesh@511",
            "cookies_file": "/app/cookies/twitter_account1.json"
        },
        {
            "username": "RegisRefre66143",
            "email": "regis.refren789@gmail.com",
            "password": "Maneesh@789",
            "cookies_file": "/app/cookies/twitter_account2.json"
        }
    ]
    # count = int(os.getenv("TWITTER_ACCOUNTS_COUNT", "1"))
    # for i in range(1, count + 1):
    #     username = os.getenv(f"TWITTER_USERNAME_{i}")
    #     email = os.getenv(f"TWITTER_EMAIL_{i}")
    #     password = os.getenv(f"TWITTER_PASSWORD_{i}")
    #     cookies_file = os.getenv(f"TWITTER_COOKIES_FILE_{i}", f"/app/cookies/twitter_account{i}.json")
    #     if username and email and password:
    #         accounts.append({
    #             "username": username,
    #             "email": email,
    #             "password": password,
    #             "cookies_file": cookies_file
    #         })
    logger.info(accounts)
    return accounts 