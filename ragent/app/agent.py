from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import List, Dict, Optional, TypedDict
import os
from dotenv import load_dotenv
from urllib.parse import urljoin
import re
from tenacity import retry, stop_after_attempt, wait_fixed
import structlog
import json
import praw
from datetime import datetime, timedelta
import asyncio
from app.models import WebsiteDataModel, RedditPostModel
from app.database import get_db

# Initialize logging
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

# FastAPI router
router = APIRouter()

# --- Website Scraping Logic ---

class WebsiteInput(BaseModel):
    url: str

class WebsiteData(BaseModel):
    url: str
    title: str
    description: str
    target_audience: str
    keywords: List[str]
    products_services: List[Dict[str, str]]

async def analyze_content(content: str, url: str, schema_products: List[Dict]) -> tuple[str, List[str], List[Dict[str, str]]]:
    try :
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro")

def find_product_service_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    keywords = ["products", "services", "offerings", "solutions", "shop"]
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].lower()
        if any(kw in href for kw in keywords):
            full_url = urljoin(base_url, href)
            if is_valid_url(full_url):
                links.append(full_url)
    return list(set(links))[:3]

def parse_html(soup: BeautifulSoup, url: str, extra_content: str = "") -> Dict:
    title = soup.find("title").text.strip() if soup.find("title") else ""
    if not title:
        og_title = soup.find("meta", property="og:title")
        title = og_title["content"].strip() if og_title else ""
    if not title:
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        title = twitter_title["content"].strip() if twitter_title else "No title found"

    description_tag = soup.find("meta", attrs={"name": "description"})
    description = description_tag["content"].strip() if description_tag else ""
    if not description:
        og_description = soup.find("meta", property="og:description")
        description = og_description["content"].strip() if og_description else ""
    if not description:
        twitter_description = soup.find("meta", attrs={"name": "twitter:description"})
        description = twitter_description["content"].strip() if twitter_description else "No description found"

    schema_data = []
    schema_products = []
    schema_scripts = soup.find_all("script", type="application/ld+json")
    for script in schema_scripts:
        try:
            json_data = json.loads(script.text)
            if isinstance(json_data, dict) and json_data.get("@type") in ["Product", "Service"]:
                product_info = {
                    "name": json_data.get("name", ""),
                    "description": json_data.get("description", ""),
                    "price": json_data.get("offers", {}).get("price", "") if json_data.get("offers") else "",
                    "category": json_data.get("category", "")
                }
                schema_products.append(product_info)
                schema_data.append(json.dumps(json_data))
        except json.JSONDecodeError:
            logger.warning("Invalid JSON-LD", url=url)

    body = soup.find("body")
    content = body.get_text(separator=" ", strip=True)[:10000] if body else ""
    content = f"{content} {extra_content} {' '.join(schema_data)}".strip()
    obj ={"title": title, "description": description, "content": content, "schema_products": schema_products}
    print(obj)
    logger.info("Successfully parsed HTML", url=url, title=title)
    return {"title": title, "description": description, "content": content, "schema_products": schema_products}


async def scrape_additional_pages(links: List[str], session: aiohttp.ClientSession) -> str:
    extra_content = []
    for link in links:
        try:
            async with session.get(link, timeout=5) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), "html.parser")
                    body = soup.find("body")
                    if body:
                        extra_content.append(body.get_text(separator=" ", strip=True)[:5000])
                        logger.info("Scraped additional page", url=link)
        except Exception as e:
            logger.warning("Failed to scrape additional page", url=link, error=str(e))
    return " ".join(extra_content)

def is_valid_url(url: str) -> bool:
    pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    return bool(pattern.match(url))

async def get_async_session():
    return aiohttp.ClientSession(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
async def scrape_website(url:str) -> Dict:
    if not is_valid_url(url):
        logger.error("Invalid URL", url=url)
        raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
    
    async with await get_async_session() as session:
        try:
            async with session.get(url, timeout=10) as response:
                if response.status !=200:
                    logger.error("HTTP error", url=url, status=response.status)
                    raise HTTPException(status_code=response.status, detail="HTTP error")
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                product_links = find_product_service_links(soup, url)
                extra_content = await scrape_additional_pages(product_links, session)
                return parse_html(soup, url, extra_content)
            

        except aiohttp.ClientError as e:
            logger.warning("Async scrape failed, trying Selenium", url=url, error=str(e))
            return []


@router.post("/scraper/scrape-website", dependencies=[Depends(RateLimiter(times=90, seconds=60))])
async def scrape_website_data(input : WebsiteInput, db: AsyncSession =Depends(get_db)):
    if not is_valid_url(input.url):
        logger.error("Invalid URL", url=input.url)
        raise HTTPException(status_code=400, detail=f"Invalid URL format: {input.url}")

    try:
        scraped_data = await scrape_website(input.url)
        logger.info("Website processing completed", url = input.url)
    
    except Exception as e:
        logger.error("Website processing failed", url=input.url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
