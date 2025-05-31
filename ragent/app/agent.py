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
    main_category: Optional[str] = None # Added main_category field

async def store_website_results(url:str, data: WebsiteData, db: AsyncSession):
    try:
        website_data = WebsiteDataModel(
            url=data.url,
            title=data.title,
            description=data.description,
            target_audience=data.target_audience,
            keywords=data.keywords,
            products_services=data.products_services,
            main_category=data.main_category # Store main_category
        )
        async with db.begin():
            db.add(website_data)
            await db.commit()
        logger.info("Stored website results in DB", url=url)
    except IntegrityError as e:
        logger.error("Database error", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def analyze_content(content: str, url: str, schema_products: List[Dict]) -> tuple[str, str, List[str], List[Dict[str, str]], str]: # Updated return type hint
    try :
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        #Chunking
        chunk_size = 8000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        print(chunks)


        # Define prompt templates
        description_prompt = PromptTemplate(
            input_variables=["content"],
            template="""Extract a concise 8-10 sentence website description from this content.
            Focus on the core purpose or value proposition. Be specific and avoid generic phrases.
            Content: {content}"""
        )
        audience_prompt = PromptTemplate(
            input_variables=["content"],
            template="Analyze this website content chunk and identify the target audience (e.g., startups, marketers, developers) separated by commas. {content}"
        )
        keywords_prompt = PromptTemplate(
            input_variables=["content"],
            template="Extract 5-10 key topics or keywords from this content chunk, separated by commas. Focus on unique concepts: {content}"
        )
        products_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
        You are an intelligent assistant designed to extract product and service information from text.

        Your task is to carefully read the provided content and return a **JSON array** of unique products or services explicitly or implicitly mentioned.

        Each product/service should be represented as an object with the following fields:
        - "name": The clear name of the product/service.
        - "description": A short and concise description in plain English.
        - "price": The price mentioned in the text. If no price is available, use "N/A".
        - "category": The general category it belongs to (e.g., "software", "hardware", "consulting service", "SaaS", "API", "platform", etc.)

        ### Rules:
        1. Only extract **products/services** — not features, companies, or general terms.
        2. Do **not** repeat any previously mentioned products/services.
        3. If there is ambiguity (e.g. no name but a description), infer a generic name like `"Unnamed cloud backup service"` and still include it.
        4. Maintain a JSON array output only — do not include any explanation or extra text.
        {content}
            """
        )
        main_category_prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following website content and identify the single most relevant main category that describes the website's primary focus or the main product/service it offers. Provide only the category name, e.g., 'E-commerce', 'SaaS', 'Blog', 'Marketing Agency', 'Education Platform', 'Fintech', 'Healthcare', 'Portfolio', 'News', 'Community Forum', 'Consulting Services', 'Software Development', 'Hardware Products'. If unsure, provide a best guess from common website categories. Do not include any other text or explanation.
            Content: {content}"""
        )

        # Initialize LLM chains
        description_chain = LLMChain(llm=llm, prompt=description_prompt)
        audience_chain = LLMChain(llm=llm, prompt=audience_prompt)
        keywords_chain = LLMChain(llm=llm, prompt=keywords_prompt)
        products_chain = LLMChain(llm=llm, prompt=products_prompt)
        main_category_chain = LLMChain(llm=llm, prompt=main_category_prompt)

        # Prepare concurrent tasks
        tasks = [
            description_chain.arun(content=chunks[0]),
            audience_chain.arun(content=chunks[0]), # Use first chunk for audience
            main_category_chain.arun(content=chunks[0]), # Use first chunk for main category
        ]

        # Add keyword and product extraction tasks for all chunks
        keyword_tasks = [keywords_chain.arun(content=chunk) for chunk in chunks]
        product_tasks = [products_chain.ainvoke({"content": chunk}) for chunk in chunks]
        tasks.extend(keyword_tasks)
        tasks.extend(product_tasks)


        # Run tasks concurrently
        results = await asyncio.gather(*tasks)

        logger.info("Concurrent LLM tasks completed", url=url, results_type=type(results), results_length=len(results), results_sample=results[:5]) # Log type and length and sample

        # Process results - account for the first 3 results being description, audience, main_category
        description = results.pop(0).strip()
        target_audience = results.pop(0).strip()
        main_category = results.pop(0).strip()

        logger.info("Initial results processed", url=url, description=description, target_audience=target_audience, main_category=main_category)

        # The remaining results are keyword and product results, interleaved (keywords first, then products)
        keyword_results = results[:len(chunks)]
        product_results = results[len(chunks):]

        logger.info("Separated keyword and product results", url=url, keyword_results_count=len(keyword_results), product_results_count=len(product_results), keyword_results_type=type(keyword_results), product_results_type=type(product_results))

        # Process keywords
        keyword_sets = []
        for chunk_keywords_str in keyword_results:
             # Add logging for keyword processing input
             logger.info("Processing keyword result chunk", url=url, chunk_keywords_str_type=type(chunk_keywords_str), chunk_keywords_str_sample=chunk_keywords_str[:100])
             chunk_keywords = chunk_keywords_str.split(", ")
             keyword_sets.append(set(kw.strip().lower() for kw in chunk_keywords))

        combined_keywords = set()
        for kw_set in keyword_sets:
            combined_keywords.update(kw_set)
        final_keywords = list(combined_keywords)  # We can take top 10 most frequent if needed

        print(final_keywords)

        # Process products
        all_products = schema_products.copy()
        seen_products = set(p["name"].lower() for p in schema_products)
        for raw_response in product_results:
             try:
                # Assuming raw_response is the dictionary returned by ainvoke
                # Add logging for raw_response before processing
                logger.info("Processing product result", url=url, raw_response_type=type(raw_response), raw_response_keys=list(raw_response.keys()) if isinstance(raw_response, dict) else None, raw_response_sample=str(raw_response)[:100])
                json_string = re.sub(r'^```json|```$', '', raw_response['text'].strip()).strip()
                chunk_products = json.loads(json_string)
                for product in chunk_products:
                    if product["name"].lower() not in seen_products:
                        all_products.append(product)
                        seen_products.add(product["name"].lower())
             except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Product parsing failed for chunk: {str(e)}")
                # Log the raw response that caused the error
                logger.warning("Failed product raw response", url=url, raw_response=raw_response)
                continue
            
        logger.info("Content analyzed with chunking and concurrent LLM calls", 
                   url=url, 
                   chunks=len(chunks),
                   audience=target_audience, 
                   keywords=len(final_keywords),
                   products=len(all_products))


        return (
            description,
            target_audience,
            final_keywords,
            all_products,
            main_category # Added main_category to return value
        )
    except Exception as e:
        logger.error("Gemini analysis failed for URL %s with error: %s", url, str(e))
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
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
        # Updated unpacking to include main_category
        description, target_audience, keywords, products_services, main_category = await analyze_content(scraped_data["content"],input.url, scraped_data["schema_products"])
        result = WebsiteData(
            url=input.url,
            title=scraped_data["title"],
            description=description,
            target_audience=target_audience,
            keywords=keywords,
            products_services=products_services,
            main_category=main_category # Pass main_category to WebsiteData
        )
        await store_website_results(input.url, result, db)
        logger.info("Website processing completed", url = input.url)
        return result
    
    except Exception as e:
        logger.error("Website processing failed", url=input.url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

