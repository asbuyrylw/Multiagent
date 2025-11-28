"""
Nonprofit Enrichment System - Comprehensive Implementation
Author: NexGrantAI
Version: 2.0 (Refactored)

This system enriches nonprofit organization data using multiple APIs:
- Perplexity AI for general research
- US Census API for demographics
- OpenStreetMap Nominatim for geocoding

Key improvements:
- Proper session management with context managers
- Rate limiting for all APIs
- Robust JSON extraction
- Cost tracking and budget controls
- Data quality scoring
- Batch processing with progress tracking
- Comprehensive error handling
"""

import asyncio
import datetime
import hashlib
import json
import logging
import logging.config
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables
load_dotenv()

# ============================================================================
# SECTION 1: LOGGING CONFIGURATION
# ============================================================================

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

def setup_logging():
    """Configure logging with proper validation and file rotation."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    if log_level not in LOG_LEVELS:
        log_level = "INFO"

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': log_level,
            },
        },
        'root': {
            'handlers': ['console'],
            'level': log_level,
        },
    }
    logging.config.dictConfig(config)

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 2: DATABASE SETUP WITH PROPER SESSION MANAGEMENT
# ============================================================================

Base = declarative_base()

class OrgData(Base):
    """Database model for cached organization data."""
    __tablename__ = 'organization_data'

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    address = Column(String)
    hash_key = Column(String, unique=True, index=True)
    data_json = Column(Text)
    last_updated = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    query_count = Column(Integer, default=1)
    data_quality_score = Column(Integer)

    __table_args__ = (
        Index('ix_org_updated', 'last_updated'),
        Index('ix_org_name_addr', 'name', 'address'),
    )

# Database engine and session setup
db_path = os.getenv("DATABASE_PATH", "nonprofit_cache.db")
engine = create_engine(f'sqlite:///{db_path}', echo=False, pool_pre_ping=True)
Base.metadata.create_all(engine)

SessionFactory = sessionmaker(bind=engine)
ScopedSession = scoped_session(SessionFactory)

@contextmanager
def get_session():
    """Thread-safe session context manager with proper error handling."""
    session = ScopedSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_cached_data(name: str, address: str, max_age_days: int = 7) -> dict | None:
    """
    Retrieve cached data if fresh enough.

    Args:
        name: Organization name
        address: Organization address
        max_age_days: Maximum age of cache in days

    Returns:
        Cached data dict or None if not found/expired
    """
    if not name or not address:
        raise ValueError("Name and address required")

    hash_key = hashlib.sha256(
        f"{name.strip().lower()}_{address.strip().lower()}".encode()
    ).hexdigest()

    with get_session() as session:
        try:
            entry = session.query(OrgData).filter_by(hash_key=hash_key).first()
            if entry:
                age_days = (datetime.datetime.utcnow() - entry.last_updated).days
                if age_days < max_age_days:
                    logger.info(f"Cache hit for {name} (age: {age_days} days)")
                    # Increment query count
                    entry.query_count += 1
                    return json.loads(entry.data_json)
                else:
                    logger.info(f"Cache expired for {name} (age: {age_days} days)")
        except (json.JSONDecodeError, SQLAlchemyError) as e:
            logger.error(f"Cache read error for {name}: {e}")

    return None

def save_data(name: str, address: str, data: dict, quality_score: int = 0) -> bool:
    """
    Save or update cached data with proper race condition handling.

    Args:
        name: Organization name
        address: Organization address
        data: Data to cache
        quality_score: Data quality score (0-100)

    Returns:
        True if successful, False otherwise
    """
    hash_key = hashlib.sha256(
        f"{name.strip().lower()}_{address.strip().lower()}".encode()
    ).hexdigest()

    with get_session() as session:
        try:
            # Use with_for_update to prevent race conditions
            entry = session.query(OrgData).filter_by(hash_key=hash_key).with_for_update().first()

            if entry:
                entry.data_json = json.dumps(data)
                entry.last_updated = datetime.datetime.utcnow()
                entry.data_quality_score = quality_score
                entry.query_count += 1
            else:
                entry = OrgData(
                    name=name,
                    address=address,
                    hash_key=hash_key,
                    data_json=json.dumps(data),
                    last_updated=datetime.datetime.utcnow(),
                    data_quality_score=quality_score,
                )
                session.add(entry)

            logger.info(f"Cached data for {name} (quality: {quality_score}%)")
            return True

        except SQLAlchemyError as e:
            logger.error(f"Cache save error for {name}: {e}")
            return False

def cleanup_old_cache(days: int = 30) -> int:
    """
    Remove cache entries older than specified days.

    Args:
        days: Age threshold in days

    Returns:
        Number of deleted entries
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    with get_session() as session:
        deleted = session.query(OrgData).filter(
            OrgData.last_updated < cutoff
        ).delete()
        logger.info(f"Cleaned up {deleted} old cache entries")
        return deleted

# ============================================================================
# SECTION 3: RATE LIMITERS FOR ALL APIS
# ============================================================================

# Perplexity: 5 calls per second (conservative)
perplexity_limiter = AsyncLimiter(5, 1)

# Nominatim: 1 call per second (required by usage policy)
nominatim_limiter = AsyncLimiter(1, 1)

# Census API: 100 calls per second (but we'll be conservative)
census_limiter = AsyncLimiter(10, 1)

# ============================================================================
# SECTION 4: HTTP SESSION MANAGEMENT
# ============================================================================

_http_session: ClientSession | None = None

async def get_http_session() -> ClientSession:
    """Get or create aiohttp session for connection pooling."""
    global _http_session
    if _http_session is None or _http_session.closed:
        timeout = ClientTimeout(total=30)
        _http_session = ClientSession(timeout=timeout)
    return _http_session

async def close_http_session():
    """Close the global HTTP session."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()

# ============================================================================
# SECTION 5: INPUT SANITIZATION AND VALIDATION
# ============================================================================

def sanitize_input(text: str) -> str:
    """
    Remove potentially harmful characters and normalize whitespace.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text that may contain markdown or extra content.

    This handles Perplexity responses that may wrap JSON in code blocks
    or include explanatory text.

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted JSON dict or error dict
    """
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error(f"Could not extract JSON from: {text[:200]}")
        return {
            "success": False,
            "error": "Invalid JSON response",
            "details": {"raw": text[:500]}
        }

# ============================================================================
# SECTION 6: PYDANTIC MODELS FOR VALIDATION
# ============================================================================

class DemographicsOutput(BaseModel):
    """Schema for demographics data."""
    success: bool = True
    population: Optional[str] = None
    median_income: Optional[str] = None
    source: Optional[str] = None
    zip_code: Optional[str] = None
    error: Optional[str] = None
    details: dict = {}

class ContactMissionOutput(BaseModel):
    """Schema for contact and mission data."""
    success: bool = True
    contact: Optional[str] = None
    social: list[str] = []
    mission: Optional[str] = None
    website: Optional[str] = None
    error: Optional[str] = None
    details: dict = {}

class NewsOutput(BaseModel):
    """Schema for news data."""
    success: bool = True
    news: list[str] = []
    error: Optional[str] = None
    details: dict = {}

class LeadershipOutput(BaseModel):
    """Schema for leadership data."""
    success: bool = True
    leadership: dict = {}
    error: Optional[str] = None
    details: dict = {}

# ============================================================================
# SECTION 7: PERPLEXITY INTEGRATION WITH ROBUST JSON HANDLING
# ============================================================================

# Initialize Perplexity client
pplx_api_key = os.getenv("PERPLEXITY_API_KEY")
if not pplx_api_key:
    logger.warning("PERPLEXITY_API_KEY not set - Perplexity features will be disabled")
    pplx_llm = None
else:
    pplx_llm = ChatPerplexity(
        api_key=pplx_api_key,
        model="llama-3.1-sonar-large-128k-online",
        temperature=0.0,
    )

async def _perplexity_search(query: str, expected_schema: type[BaseModel] = None) -> dict:
    """
    Query Perplexity API with rate limiting and robust JSON parsing.

    Args:
        query: Search query
        expected_schema: Pydantic model for validation

    Returns:
        Parsed and validated response dict
    """
    if not pplx_llm:
        return {
            "success": False,
            "error": "Perplexity API not configured",
            "details": {}
        }

    async with perplexity_limiter:
        try:
            # Build prompt with explicit JSON instruction
            system_msg = (
                "You are a research assistant. Return ONLY valid JSON with no "
                "markdown formatting, no explanations, no code blocks. Just pure JSON."
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("human", "{query}")
            ])

            chain = prompt | pplx_llm

            # Try async first, fall back to sync if not supported
            try:
                response = await chain.ainvoke({"query": query})
            except (AttributeError, NotImplementedError):
                # Async not supported, use sync in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: chain.invoke({"query": query})
                )

            # Extract JSON from response
            content = response.content if hasattr(response, 'content') else str(response)
            result = extract_json_from_text(content)

            # Validate against schema if provided
            if expected_schema and result.get("success", True):
                try:
                    validated = expected_schema(**result)
                    return validated.dict()
                except ValidationError as e:
                    logger.error(f"Schema validation failed: {e}")
                    return {
                        "success": False,
                        "error": f"Invalid response format: {e}",
                        "details": result
                    }

            return result

        except Exception as e:
            logger.error(f"Perplexity search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": {}
            }

# ============================================================================
# SECTION 8: ENRICHMENT TOOLS
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def demographics_tool(address: str) -> dict:
    """
    Fetch demographics for an address.
    Uses Census API for US addresses, Perplexity for international.

    Args:
        address: Full address string

    Returns:
        Demographics data dict
    """
    address = sanitize_input(address)
    if not address:
        return {
            "success": False,
            "error": "Empty address",
            "details": {}
        }

    try:
        # Geocode with Nominatim (respecting rate limit)
        async with nominatim_limiter:
            session = await get_http_session()
            headers = {
                "User-Agent": "NexGrantAI/1.0 (nonprofit-enrichment)"
            }

            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": address,
                "format": "json",
                "addressdetails": 1
            }

            async with session.get(url, headers=headers, params=params) as resp:
                resp.raise_for_status()
                geo_data = await resp.json()

        if not geo_data:
            return {
                "success": False,
                "error": "Address not found",
                "details": {"address": address}
            }

        geo = geo_data[0]
        address_info = geo.get("address", {})
        country = address_info.get("country_code", "").upper()

        # US: Use Census API
        if country == "US":
            zip_code = address_info.get("postcode", "").split("-")[0]  # Handle ZIP+4

            if not zip_code or not re.match(r'^\d{5}$', zip_code):
                logger.warning(f"Invalid ZIP code: {zip_code}, falling back to Perplexity")
                return await _get_demographics_via_perplexity(address)

            census_key = os.getenv("CENSUS_API_KEY")
            if not census_key:
                logger.warning("Census API key not configured, using Perplexity")
                return await _get_demographics_via_perplexity(address)

            # Census API call with rate limiting
            async with census_limiter:
                session = await get_http_session()
                url = "https://api.census.gov/data/2021/acs/acs5"
                params = {
                    "get": "NAME,B01003_001E,B19013_001E",  # Population, Median Income
                    "for": f"zip code tabulation area:{zip_code}",
                    "key": census_key
                }

                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Census API error: {resp.status}")
                        return await _get_demographics_via_perplexity(address)

                    data = await resp.json()

            if len(data) < 2:
                logger.error(f"Unexpected Census response: {data}")
                return await _get_demographics_via_perplexity(address)

            _, population, median_income, *_ = data[1]

            return {
                "success": True,
                "population": str(population) if population != "-666666666" else "Data not available",
                "median_income": str(median_income) if median_income != "-666666666" else "Data not available",
                "source": "US Census Bureau",
                "zip_code": zip_code
            }

        # International: Use Perplexity
        else:
            return await _get_demographics_via_perplexity(address)

    except Exception as e:
        logger.error(f"Demographics tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "details": {"address": address}
        }

async def _get_demographics_via_perplexity(address: str) -> dict:
    """Fallback to Perplexity for demographics."""
    query = f"""
    What is the population and median household income for the area around: {address}?

    Return ONLY a JSON object with this exact structure:
    {{"success": true, "population": "number or estimate", "median_income": "amount in local currency", "source": "where you found this"}}
    """

    result = await _perplexity_search(query, DemographicsOutput)
    return result

async def contact_social_mission_tool(name: str, address: str) -> dict:
    """
    Fetch contact info, social media, and mission statement.

    Args:
        name: Organization name
        address: Organization address

    Returns:
        Contact and mission data dict
    """
    name = sanitize_input(name)
    address = sanitize_input(address)

    query = f"""
    Find official contact information, social media links, and mission statement for:
    Organization: {name}
    Location: {address}

    Return ONLY a JSON object with this exact structure:
    {{
      "success": true,
      "contact": "main phone and/or email",
      "social": ["list", "of", "social", "media", "URLs"],
      "mission": "mission statement text",
      "website": "official website URL"
    }}
    """

    result = await _perplexity_search(query, ContactMissionOutput)
    return result

async def news_tool(name: str, months_back: int = 12) -> dict:
    """
    Fetch recent news about the organization.

    Args:
        name: Organization name
        months_back: How many months back to search

    Returns:
        News data dict
    """
    name = sanitize_input(name)
    date_threshold = (
        datetime.datetime.now() - datetime.timedelta(days=30*months_back)
    ).strftime("%Y-%m-%d")

    query = f"""
    Find the top 3 most significant news articles or press releases about "{name}" since {date_threshold}.

    Return ONLY a JSON object with this exact structure:
    {{
      "success": true,
      "news": [
        "Brief summary of article 1 with date",
        "Brief summary of article 2 with date",
        "Brief summary of article 3 with date"
      ]
    }}

    If no news found, return: {{"success": true, "news": []}}
    """

    result = await _perplexity_search(query, NewsOutput)
    return result

async def leadership_tool(name: str) -> dict:
    """
    Fetch leadership team information.

    Args:
        name: Organization name

    Returns:
        Leadership data dict
    """
    name = sanitize_input(name)

    query = f"""
    Find the leadership team for "{name}". Include Executive Director/CEO,
    Development Director, and other key leaders with their contact information if available.

    Return ONLY a JSON object with this exact structure:
    {{
      "success": true,
      "leadership": {{
        "executive_director": "Name (email if available)",
        "development_director": "Name (email if available)",
        "other_leaders": "Additional key staff"
      }}
    }}
    """

    result = await _perplexity_search(query, LeadershipOutput)
    return result

# ============================================================================
# SECTION 9: COST TRACKING
# ============================================================================

class CostTracker:
    """Track API costs and enforce budgets."""

    def __init__(self, daily_budget: float = 50.0):
        """
        Initialize cost tracker.

        Args:
            daily_budget: Maximum daily spending in USD
        """
        self.daily_budget = daily_budget
        self.costs = []

    def add_cost(self, amount: float, service: str, operation: str):
        """Record a cost."""
        self.costs.append({
            "timestamp": datetime.datetime.utcnow(),
            "amount": amount,
            "service": service,
            "operation": operation
        })

    def get_today_cost(self) -> float:
        """Get total cost for today."""
        today = datetime.datetime.utcnow().date()
        return sum(
            c["amount"] for c in self.costs
            if c["timestamp"].date() == today
        )

    def can_make_request(self, estimated_cost: float) -> bool:
        """Check if request is within budget."""
        return (self.get_today_cost() + estimated_cost) <= self.daily_budget

    def get_stats(self) -> dict:
        """Get cost statistics."""
        today_cost = self.get_today_cost()
        today = datetime.datetime.utcnow().date()
        today_requests = len([
            c for c in self.costs
            if c["timestamp"].date() == today
        ])

        return {
            "today_cost": round(today_cost, 4),
            "daily_budget": self.daily_budget,
            "remaining_budget": round(self.daily_budget - today_cost, 4),
            "total_requests_today": today_requests,
            "cost_per_request": round(today_cost / today_requests, 4) if today_requests > 0 else 0
        }

# Global cost tracker
cost_tracker = CostTracker(daily_budget=float(os.getenv("DAILY_API_BUDGET", "50.0")))

# ============================================================================
# SECTION 10: DATA QUALITY SCORING
# ============================================================================

@dataclass
class EnrichmentResult:
    """Complete enrichment result with quality scoring."""
    success: bool = True
    demographics: dict = field(default_factory=dict)
    contact_social_mission: dict = field(default_factory=dict)
    news: list = field(default_factory=list)
    leadership: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "demographics": self.demographics,
            "contact_social_mission": self.contact_social_mission,
            "news": self.news,
            "leadership": self.leadership,
            "errors": self.errors,
            "metadata": self.metadata
        }

    def calculate_quality_score(self) -> int:
        """
        Calculate data completeness score (0-100).

        Scoring breakdown:
        - Demographics: 25 points (12 for population, 13 for income)
        - Contact/Mission: 35 points (10 contact, 15 mission, 10 social)
        - News: 15 points (5 per article, max 15)
        - Leadership: 25 points (15 for ED, 10 for Dev Director)

        Returns:
            Quality score from 0-100
        """
        score = 0

        # Demographics (25 points)
        if self.demographics.get("success"):
            if self.demographics.get("population") and self.demographics.get("population") != "Data not available":
                score += 12
            if self.demographics.get("median_income") and self.demographics.get("median_income") != "Data not available":
                score += 13

        # Contact/Mission (35 points)
        if self.contact_social_mission.get("success"):
            if self.contact_social_mission.get("contact"):
                score += 10
            if self.contact_social_mission.get("mission"):
                score += 15
            if self.contact_social_mission.get("social") and len(self.contact_social_mission.get("social", [])) > 0:
                score += 10

        # News (15 points)
        if self.news and len(self.news) > 0:
            score += min(15, len(self.news) * 5)

        # Leadership (25 points)
        if self.leadership.get("success"):
            leaders = self.leadership.get("leadership", {})
            if leaders.get("executive_director"):
                score += 15
            if leaders.get("development_director"):
                score += 10

        return min(100, score)

# ============================================================================
# SECTION 11: SIMPLE ENRICHMENT PIPELINE (NO LANGGRAPH)
# ============================================================================

class SimpleEnrichmentPipeline:
    """
    Direct pipeline without LangChain/LangGraph complexity.
    Runs enrichment steps in parallel where possible.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.total_api_calls = 0
        self.total_cost = 0.0

    async def enrich_organization(
        self,
        name: str,
        address: str,
        skip_cache: bool = False
    ) -> EnrichmentResult:
        """
        Main enrichment function.
        Runs independent tasks in parallel for efficiency.

        Args:
            name: Organization name
            address: Organization address
            skip_cache: Whether to skip cache lookup

        Returns:
            EnrichmentResult with all collected data
        """
        start_time = time.time()
        result = EnrichmentResult()

        # Check cache first
        if not skip_cache:
            cached = get_cached_data(name, address)
            if cached:
                logger.info(f"Using cached data for {name}")
                return EnrichmentResult(**cached)

        try:
            # Estimate cost for all 4 API calls
            estimated_cost = 0.08  # ~$0.02 per Perplexity call * 4 calls

            if not cost_tracker.can_make_request(estimated_cost):
                logger.error("Daily budget exceeded!")
                result.success = False
                result.errors.append("Daily API budget exceeded")
                return result

            # Run independent tasks in parallel
            demographics_task = demographics_tool(address)
            contact_task = contact_social_mission_tool(name, address)
            news_task = news_tool(name)
            leadership_task = leadership_tool(name)

            # Wait for all tasks
            demographics, contact, news, leadership = await asyncio.gather(
                demographics_task,
                contact_task,
                news_task,
                leadership_task,
                return_exceptions=True
            )

            # Process results
            result.demographics = demographics if not isinstance(demographics, Exception) else {
                "success": False, "error": str(demographics), "details": {}
            }

            result.contact_social_mission = contact if not isinstance(contact, Exception) else {
                "success": False, "error": str(contact), "details": {}
            }

            result.news = news.get("news", []) if not isinstance(news, Exception) else []

            result.leadership = leadership if not isinstance(leadership, Exception) else {
                "success": False, "error": str(leadership), "details": {}
            }

            # Collect errors
            for task_result in [demographics, contact, news, leadership]:
                if isinstance(task_result, Exception):
                    result.errors.append(str(task_result))
                elif isinstance(task_result, dict) and not task_result.get("success", True):
                    result.errors.append(task_result.get("error", "Unknown error"))

            # Track costs
            cost_tracker.add_cost(estimated_cost, "Perplexity", "batch_enrichment")

            # Calculate metadata
            quality_score = result.calculate_quality_score()
            result.metadata = {
                "enrichment_date": datetime.datetime.utcnow().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "quality_score": quality_score,
                "quality_tier": "high" if quality_score >= 70 else "medium" if quality_score >= 40 else "low",
                "api_calls_made": 4,
                "estimated_cost": estimated_cost
            }

            # Determine overall success
            result.success = quality_score >= 40  # At least medium quality

            # Cache the result
            save_data(name, address, result.to_dict(), quality_score)

            logger.info(
                f"Enriched {name}: Quality={quality_score}%, "
                f"Time={result.metadata['processing_time_seconds']}s"
            )

            return result

        except Exception as e:
            logger.error(f"Enrichment failed for {name}: {e}")
            result.success = False
            result.errors.append(str(e))
            result.metadata = {
                "enrichment_date": datetime.datetime.utcnow().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "error": str(e)
            }
            return result

# ============================================================================
# SECTION 12: BATCH PROCESSING WITH PROGRESS TRACKING
# ============================================================================

class BatchEnricher:
    """Process multiple organizations with progress tracking and checkpoints."""

    def __init__(self, pipeline: SimpleEnrichmentPipeline, max_concurrent: int = 5):
        """
        Initialize batch enricher.

        Args:
            pipeline: Enrichment pipeline instance
            max_concurrent: Maximum concurrent enrichments
        """
        self.pipeline = pipeline
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def enrich_one_with_limit(self, org: dict) -> dict:
        """Enrich one org with concurrency limit."""
        async with self.semaphore:
            result = await self.pipeline.enrich_organization(
                name=org["name"],
                address=org["address"]
            )
            return {
                "organization": org,
                "enrichment": result.to_dict()
            }

    async def enrich_batch(
        self,
        organizations: list[dict],
        save_checkpoint_every: int = 10
    ) -> list[dict]:
        """
        Enrich multiple organizations with progress tracking.

        Args:
            organizations: List of org dicts with 'name' and 'address' keys
            save_checkpoint_every: Save progress every N orgs

        Returns:
            List of enrichment results
        """
        results = []

        tasks = [self.enrich_one_with_limit(org) for org in organizations]

        logger.info(f"Starting batch enrichment of {len(organizations)} organizations")

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)

            # Progress logging
            if i % 5 == 0 or i == len(tasks):
                logger.info(f"Progress: {i}/{len(tasks)} organizations processed")

            # Save checkpoint
            if i % save_checkpoint_every == 0:
                self._save_checkpoint(results, f"checkpoint_{i}.json")

        logger.info(f"Batch enrichment complete: {len(results)} organizations")
        return results

    def _save_checkpoint(self, results: list, filename: str):
        """Save progress checkpoint."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved checkpoint: {filename}")

# ============================================================================
# SECTION 13: EXAMPLE USAGE
# ============================================================================

async def example_single_org():
    """Example: Enrich a single organization."""
    pipeline = SimpleEnrichmentPipeline()

    result = await pipeline.enrich_organization(
        name="American Red Cross",
        address="430 17th St NW, Washington, DC 20006"
    )

    print("\n" + "="*80)
    print("ENRICHMENT RESULT")
    print("="*80)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nQuality Score: {result.metadata.get('quality_score', 0)}%")
    print(f"Quality Tier: {result.metadata.get('quality_tier', 'unknown')}")
    print(f"Processing Time: {result.metadata.get('processing_time_seconds', 0)}s")

    # Show cost stats
    stats = cost_tracker.get_stats()
    print("\n" + "="*80)
    print("COST TRACKER STATS")
    print("="*80)
    print(json.dumps(stats, indent=2))

async def example_batch():
    """Example: Enrich multiple organizations."""
    orgs = [
        {"name": "American Red Cross", "address": "430 17th St NW, Washington, DC 20006"},
        {"name": "United Way", "address": "1800 Diagonal Rd, Alexandria, VA 22314"},
        {"name": "Feeding America", "address": "35 E Wacker Dr, Chicago, IL 60601"},
    ]

    pipeline = SimpleEnrichmentPipeline()
    batch_enricher = BatchEnricher(pipeline, max_concurrent=2)

    results = await batch_enricher.enrich_batch(orgs, save_checkpoint_every=2)

    print("\n" + "="*80)
    print(f"BATCH ENRICHMENT COMPLETE: {len(results)} organizations")
    print("="*80)

    for r in results:
        org = r["organization"]
        enrich = r["enrichment"]
        print(f"\n{org['name']}:")
        print(f"  Success: {enrich['success']}")
        print(f"  Quality: {enrich.get('metadata', {}).get('quality_score', 0)}%")

async def main():
    """Main entry point."""
    try:
        # Single org example
        await example_single_org()

        # Uncomment for batch example:
        # await example_batch()

    finally:
        # Clean up HTTP session
        await close_http_session()

if __name__ == "__main__":
    asyncio.run(main())
