import asyncio
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .models import AgentState
from twikit import TooManyRequests
from .x_agent_config import logger, MIN_WAIT_TIME, MAX_WAIT_TIME
from .client import get_twitter_client
from .models import TweetQueryInput
from .x_agent_config import load_twitter_accounts_from_env

def extract_hashtags_from_text(text: str) -> List[str]:
    """Extract hashtags from tweet text"""
    return [word for word in text.split() if word.startswith('#')] 

async def get_tweets(client, query: str, tweets=None):
    """Get tweets with rate limiting and pagination handling"""
    try:
        if tweets is None:
            logger.info(f"Getting initial tweets for query: {query}")
            tweets = await client.search_tweet(query, 'Latest')
        else:
            wait_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
            logger.info(f"Getting next tweets after {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            tweets = await tweets.next()
        return tweets
    except TooManyRequests as e:
        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
        wait_time = (rate_limit_reset - datetime.now(timezone.utc)).total_seconds()
        wait_time = max(wait_time + 60, 600)  # Minimum 10 minutes wait
        logger.warning(f"Rate limit reached. Waiting until {rate_limit_reset}", 
                      wait_seconds=wait_time)
        await asyncio.sleep(wait_time)
        return await get_tweets(client, query, tweets)
    except Exception as e:
        logger.error(f"Error getting tweets: {str(e)}")
        raise

async def fetch_tweets_by_query(input: TweetQueryInput, db, twitter_client=None) -> Dict:
    """Fetch tweets by query with rate limiting and pagination"""
    try:
        logger.info("Starting tweet fetch process", 
                   query=input.query,
                   minimum_tweets=input.minimum_tweets,
                   product=input.product)

        if twitter_client is None:
            from .x_agent_config import TWITTER_USERNAME, TWITTER_EMAIL, TWITTER_PASSWORD
            twitter_client = await get_twitter_client(TWITTER_USERNAME, TWITTER_EMAIL, TWITTER_PASSWORD, '/app/cookies/twitter_cookies.json')
        
        tweet_count = 0
        tweets = None
        results = []

        while tweet_count < input.minimum_tweets:
            try:
                if tweets is None:
                    logger.info(f"Getting initial tweets for query: {input.query}")
                    tweets = await twitter_client.search_tweet(input.query, product=input.product)
                else:
                    wait_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
                    logger.info(f"Getting next tweets after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    tweets = await tweets.next()

            except TooManyRequests as e:
                rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                wait_time = (rate_limit_reset - datetime.now(timezone.utc)).total_seconds()
                wait_time = max(wait_time + 60, 600)  # Minimum 10 minutes wait
                logger.warning(f"Rate limit reached. Waiting until {rate_limit_reset}", 
                             wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                continue

            if not tweets:
                logger.info("No more tweets found")
                break

            for tweet in tweets:
                logger.info(tweet.id)
                tweet_count += 1
                hashtags = extract_hashtags_from_text(tweet.text)
                tweet_data = {
                    "tweet_id": tweet.id,
                    "tweet_count": tweet_count,
                    "username": tweet.user.name,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "retweets": tweet.retweet_count,
                    "likes": tweet.favorite_count,
                    "hashtags": hashtags
                }
                results.append(tweet_data)

            logger.info(f"Got {tweet_count} tweets")

        logger.info("Tweet fetch completed", 
                   total_tweets=tweet_count,
                   query=input.query)

        return {
            "status": "success",
            "total_tweets": tweet_count,
            "tweets": results
        }

    except Exception as e:
        logger.error("Tweet fetch failed", 
                    error=str(e),
                    stack_info=True)
        raise Exception(f"Error fetching tweets: {str(e)}") 
    

async def fetch_tweets_for_hashtag(client, hashtag, db):
    try:
        query_input = TweetQueryInput(
            query=f"#{hashtag}",
            minimum_tweets=10,
            product="Top"
        )
        result = await fetch_tweets_by_query(query_input, db, twitter_client=client)
        return result.get("tweets", [])
    except Exception as e:
        logger.error(f"Error processing hashtag {hashtag}", error=str(e), stack_info=True)
        return []
    
# accounts = load_twitter_accounts_from_env()
async def fetch_tweets_node(state: AgentState) -> AgentState:
    accounts = load_twitter_accounts_from_env()
    if state.get("error"):
        return state

    start_time = datetime.now(timezone.utc)
    state["tweets"] = []

    # Create clients up front, one per account
    clients = []
    for account in accounts:
        client = await get_twitter_client(
            account["username"],
            account["email"],
            account["password"],
            account["cookies_file"]
        )
        clients.append(client)

    hashtags = state["hashtags"]
    batch_size = min(3, len(clients))  # 3 or number of accounts, whichever is smaller

    for batch_start in range(0, len(hashtags), batch_size):
        batch = hashtags[batch_start:batch_start + batch_size]
        tasks = []
        for i, hashtag in enumerate(batch):
            client = clients[i]  # Each hashtag in the batch gets a unique client
            tasks.append(fetch_tweets_for_hashtag(client, hashtag, state["db"]))
        batch_results = await asyncio.gather(*tasks)
        for tweets in batch_results:
            state["tweets"].extend(tweets)
        await asyncio.sleep(random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME))

    logger.info(
        "Tweets fetched successfully",
        agent_name=state["agent_name"],
        tweet_count=len(state["tweets"]),
        duration_sec=(datetime.now(timezone.utc) - start_time).total_seconds(),
        hashtags_processed=len(state["hashtags"])
    )

    return state 