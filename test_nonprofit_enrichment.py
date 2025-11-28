"""
Test Suite for Nonprofit Enrichment System

This test suite uses mocked API responses to test the enrichment system
without making actual API calls or incurring costs.

Run with: pytest test_nonprofit_enrichment.py -v
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# Import the enrichment system
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from nonprofit_enrichment_system import (
    SimpleEnrichmentPipeline,
    EnrichmentResult,
    CostTracker,
    BatchEnricher,
    sanitize_input,
    extract_json_from_text,
    get_cached_data,
    save_data,
    demographics_tool,
    contact_social_mission_tool,
    news_tool,
    leadership_tool,
)

# ============================================================================
# MOCK DATA
# ============================================================================

MOCK_GEOCODING_RESPONSE = [{
    "address": {
        "country_code": "us",
        "postcode": "20006",
        "county": "Washington",
        "state": "DC"
    },
    "lat": "38.9072",
    "lon": "-77.0369"
}]

MOCK_CENSUS_RESPONSE = [
    ["NAME", "B01003_001E", "B19013_001E", "zip code tabulation area"],
    ["ZCTA5 20006", "25000", "95000", "20006"]
]

MOCK_PERPLEXITY_DEMOGRAPHICS = {
    "success": True,
    "population": "25000",
    "median_income": "$95,000",
    "source": "US Census Bureau via Perplexity"
}

MOCK_PERPLEXITY_CONTACT = {
    "success": True,
    "contact": "info@americanredcross.org, (202) 303-5000",
    "social": [
        "https://twitter.com/RedCross",
        "https://facebook.com/AmericanRedCross",
        "https://linkedin.com/company/american-red-cross"
    ],
    "mission": "The American Red Cross prevents and alleviates human suffering in the face of emergencies by mobilizing the power of volunteers and the generosity of donors.",
    "website": "https://www.redcross.org"
}

MOCK_PERPLEXITY_NEWS = {
    "success": True,
    "news": [
        "American Red Cross launches new blood donation campaign - March 2025",
        "Red Cross responds to major flooding in Midwest - February 2025",
        "Partnership announced with tech companies for disaster relief - January 2025"
    ]
}

MOCK_PERPLEXITY_LEADERSHIP = {
    "success": True,
    "leadership": {
        "executive_director": "Gail McGovern, President & CEO (gail.mcgovern@redcross.org)",
        "development_director": "John Smith, Chief Development Officer",
        "other_leaders": "Jane Doe (COO), Mike Johnson (CFO)"
    }
}

# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

def test_sanitize_input():
    """Test input sanitization."""
    # Test normal input
    assert sanitize_input("American Red Cross") == "American Red Cross"

    # Test with extra whitespace
    assert sanitize_input("  American   Red   Cross  ") == "American Red Cross"

    # Test with control characters
    assert sanitize_input("American\x00Red\x1FCross") == "AmericanRedCross"

    # Test empty input
    assert sanitize_input("") == ""
    assert sanitize_input(None) == ""

def test_extract_json_from_text():
    """Test JSON extraction from various formats."""
    # Test direct JSON
    result = extract_json_from_text('{"success": true, "data": "test"}')
    assert result["success"] is True
    assert result["data"] == "test"

    # Test JSON in markdown code block
    markdown_text = """
    Here's the result:
    ```json
    {"success": true, "value": 42}
    ```
    """
    result = extract_json_from_text(markdown_text)
    assert result["success"] is True
    assert result["value"] == 42

    # Test JSON without markdown tag
    markdown_text = """
    Result:
    ```
    {"success": true, "count": 10}
    ```
    """
    result = extract_json_from_text(markdown_text)
    assert result["success"] is True
    assert result["count"] == 10

    # Test JSON with surrounding text
    text = 'The answer is {"success": true, "answer": "yes"} as shown above'
    result = extract_json_from_text(text)
    assert result["success"] is True

    # Test invalid JSON
    result = extract_json_from_text("This is not JSON at all")
    assert result["success"] is False
    assert "error" in result

def test_quality_score_calculation():
    """Test quality score calculation logic."""
    # Perfect score
    result = EnrichmentResult(
        demographics={"success": True, "population": "25000", "median_income": "95000"},
        contact_social_mission={
            "success": True,
            "contact": "test@test.org",
            "social": ["https://twitter.com/test"],
            "mission": "Test mission"
        },
        news=["News 1", "News 2", "News 3"],
        leadership={
            "success": True,
            "leadership": {
                "executive_director": "John Doe",
                "development_director": "Jane Smith"
            }
        }
    )
    score = result.calculate_quality_score()
    assert score == 100

    # Partial data
    result = EnrichmentResult(
        demographics={"success": True, "population": "25000"},
        contact_social_mission={"success": True, "contact": "test@test.org"},
        news=[],
        leadership={"success": False}
    )
    score = result.calculate_quality_score()
    assert 0 < score < 50  # Should have some score but not high

    # No data
    result = EnrichmentResult(
        demographics={"success": False},
        contact_social_mission={"success": False},
        news=[],
        leadership={"success": False}
    )
    score = result.calculate_quality_score()
    assert score == 0

# ============================================================================
# DATABASE TESTS
# ============================================================================

def test_cache_save_and_retrieve():
    """Test caching functionality."""
    test_name = "Test Nonprofit"
    test_address = "123 Test St, Test City, TS 12345"
    test_data = {
        "success": True,
        "demographics": {"population": "10000"},
        "metadata": {"quality_score": 75}
    }

    # Save data
    success = save_data(test_name, test_address, test_data, quality_score=75)
    assert success is True

    # Retrieve data
    cached = get_cached_data(test_name, test_address, max_age_days=7)
    assert cached is not None
    assert cached["success"] is True
    assert cached["demographics"]["population"] == "10000"

    # Test cache expiration (mock)
    cached_fresh = get_cached_data(test_name, test_address, max_age_days=0)
    assert cached_fresh is None  # Should be expired

def test_cache_normalization():
    """Test that cache key normalization works correctly."""
    # These should all map to the same cache entry
    names = ["Test Org", "test org", "TEST ORG", "  Test   Org  "]
    addresses = ["123 Main St", "123 main st", "  123  Main  St  "]

    test_data = {"test": "data"}

    # Save with first combination
    save_data(names[0], addresses[0], test_data)

    # Retrieve with all combinations
    for name in names:
        for address in addresses:
            cached = get_cached_data(name, address)
            assert cached is not None
            assert cached["test"] == "data"

# ============================================================================
# COST TRACKER TESTS
# ============================================================================

def test_cost_tracker():
    """Test cost tracking and budget enforcement."""
    tracker = CostTracker(daily_budget=10.0)

    # Add some costs
    tracker.add_cost(2.5, "Perplexity", "search")
    tracker.add_cost(1.5, "Perplexity", "search")
    tracker.add_cost(3.0, "Census", "lookup")

    # Check total
    assert tracker.get_today_cost() == 7.0

    # Check budget remaining
    assert tracker.can_make_request(2.0) is True
    assert tracker.can_make_request(5.0) is False  # Would exceed budget

    # Check stats
    stats = tracker.get_stats()
    assert stats["today_cost"] == 7.0
    assert stats["remaining_budget"] == 3.0
    assert stats["total_requests_today"] == 3

# ============================================================================
# MOCKED API TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_demographics_tool_us_census():
    """Test demographics tool with mocked Census API."""
    with patch('nonprofit_enrichment_system.get_http_session') as mock_session:
        # Mock geocoding response
        mock_geo_response = AsyncMock()
        mock_geo_response.status = 200
        mock_geo_response.json = AsyncMock(return_value=MOCK_GEOCODING_RESPONSE)
        mock_geo_response.raise_for_status = MagicMock()

        # Mock census response
        mock_census_response = AsyncMock()
        mock_census_response.status = 200
        mock_census_response.json = AsyncMock(return_value=MOCK_CENSUS_RESPONSE)

        # Setup session mock
        mock_session_instance = AsyncMock()
        mock_session_instance.get = AsyncMock(side_effect=[
            AsyncMock(__aenter__=AsyncMock(return_value=mock_geo_response)),
            AsyncMock(__aenter__=AsyncMock(return_value=mock_census_response))
        ])
        mock_session.return_value = mock_session_instance

        # Test
        result = await demographics_tool("430 17th St NW, Washington, DC 20006")

        assert result["success"] is True
        assert result["population"] == "25000"
        assert result["median_income"] == "95000"
        assert result["source"] == "US Census Bureau"

@pytest.mark.asyncio
async def test_demographics_tool_international():
    """Test demographics tool for international address (uses Perplexity)."""
    with patch('nonprofit_enrichment_system.get_http_session') as mock_session, \
         patch('nonprofit_enrichment_system._perplexity_search') as mock_perplexity:

        # Mock geocoding for non-US
        mock_geo_response = AsyncMock()
        mock_geo_response.status = 200
        mock_geo_response.json = AsyncMock(return_value=[{
            "address": {"country_code": "fr"},
            "lat": "48.8566",
            "lon": "2.3522"
        }])
        mock_geo_response.raise_for_status = MagicMock()

        mock_session_instance = AsyncMock()
        mock_session_instance.get = AsyncMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_geo_response)
        ))
        mock_session.return_value = mock_session_instance

        # Mock Perplexity
        mock_perplexity.return_value = MOCK_PERPLEXITY_DEMOGRAPHICS

        result = await demographics_tool("Paris, France")

        assert result["success"] is True
        mock_perplexity.assert_called_once()

@pytest.mark.asyncio
async def test_contact_social_mission_tool():
    """Test contact and mission tool with mocked Perplexity."""
    with patch('nonprofit_enrichment_system._perplexity_search') as mock_perplexity:
        mock_perplexity.return_value = MOCK_PERPLEXITY_CONTACT

        result = await contact_social_mission_tool(
            "American Red Cross",
            "430 17th St NW, Washington, DC 20006"
        )

        assert result["success"] is True
        assert "info@americanredcross.org" in result["contact"]
        assert len(result["social"]) == 3
        assert "Red Cross" in result["mission"]

@pytest.mark.asyncio
async def test_news_tool():
    """Test news tool with mocked Perplexity."""
    with patch('nonprofit_enrichment_system._perplexity_search') as mock_perplexity:
        mock_perplexity.return_value = MOCK_PERPLEXITY_NEWS

        result = await news_tool("American Red Cross", months_back=6)

        assert result["success"] is True
        assert len(result["news"]) == 3
        assert "blood donation" in result["news"][0]

@pytest.mark.asyncio
async def test_leadership_tool():
    """Test leadership tool with mocked Perplexity."""
    with patch('nonprofit_enrichment_system._perplexity_search') as mock_perplexity:
        mock_perplexity.return_value = MOCK_PERPLEXITY_LEADERSHIP

        result = await leadership_tool("American Red Cross")

        assert result["success"] is True
        assert "Gail McGovern" in result["leadership"]["executive_director"]
        assert "John Smith" in result["leadership"]["development_director"]

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_simple_enrichment_pipeline():
    """Test full enrichment pipeline with all mocked APIs."""
    with patch('nonprofit_enrichment_system.demographics_tool') as mock_demo, \
         patch('nonprofit_enrichment_system.contact_social_mission_tool') as mock_contact, \
         patch('nonprofit_enrichment_system.news_tool') as mock_news, \
         patch('nonprofit_enrichment_system.leadership_tool') as mock_leadership:

        # Setup mocks
        mock_demo.return_value = MOCK_PERPLEXITY_DEMOGRAPHICS
        mock_contact.return_value = MOCK_PERPLEXITY_CONTACT
        mock_news.return_value = MOCK_PERPLEXITY_NEWS
        mock_leadership.return_value = MOCK_PERPLEXITY_LEADERSHIP

        # Run enrichment
        pipeline = SimpleEnrichmentPipeline()
        result = await pipeline.enrich_organization(
            name="American Red Cross",
            address="430 17th St NW, Washington, DC 20006",
            skip_cache=True  # Don't use cache for testing
        )

        # Verify result
        assert result.success is True
        assert result.demographics["success"] is True
        assert result.contact_social_mission["success"] is True
        assert len(result.news) == 3
        assert result.leadership["success"] is True

        # Check quality score
        quality = result.calculate_quality_score()
        assert quality >= 90  # Should have high quality with all data

        # Check metadata
        assert "enrichment_date" in result.metadata
        assert "processing_time_seconds" in result.metadata
        assert result.metadata["quality_score"] == quality

@pytest.mark.asyncio
async def test_enrichment_with_partial_failures():
    """Test enrichment when some APIs fail."""
    with patch('nonprofit_enrichment_system.demographics_tool') as mock_demo, \
         patch('nonprofit_enrichment_system.contact_social_mission_tool') as mock_contact, \
         patch('nonprofit_enrichment_system.news_tool') as mock_news, \
         patch('nonprofit_enrichment_system.leadership_tool') as mock_leadership:

        # Setup mocks with some failures
        mock_demo.return_value = {"success": False, "error": "API error", "details": {}}
        mock_contact.return_value = MOCK_PERPLEXITY_CONTACT
        mock_news.return_value = {"success": True, "news": []}
        mock_leadership.return_value = MOCK_PERPLEXITY_LEADERSHIP

        # Run enrichment
        pipeline = SimpleEnrichmentPipeline()
        result = await pipeline.enrich_organization(
            name="Test Nonprofit",
            address="Test Address",
            skip_cache=True
        )

        # Should still have partial success
        assert len(result.errors) > 0  # Has errors
        assert result.contact_social_mission["success"] is True  # But some data succeeded

        # Quality score should be medium
        quality = result.calculate_quality_score()
        assert 30 < quality < 70

@pytest.mark.asyncio
async def test_batch_enricher():
    """Test batch enrichment with progress tracking."""
    with patch('nonprofit_enrichment_system.demographics_tool') as mock_demo, \
         patch('nonprofit_enrichment_system.contact_social_mission_tool') as mock_contact, \
         patch('nonprofit_enrichment_system.news_tool') as mock_news, \
         patch('nonprofit_enrichment_system.leadership_tool') as mock_leadership:

        # Setup mocks
        mock_demo.return_value = MOCK_PERPLEXITY_DEMOGRAPHICS
        mock_contact.return_value = MOCK_PERPLEXITY_CONTACT
        mock_news.return_value = MOCK_PERPLEXITY_NEWS
        mock_leadership.return_value = MOCK_PERPLEXITY_LEADERSHIP

        # Create batch
        orgs = [
            {"name": "Org 1", "address": "Address 1"},
            {"name": "Org 2", "address": "Address 2"},
            {"name": "Org 3", "address": "Address 3"},
        ]

        # Run batch
        pipeline = SimpleEnrichmentPipeline()
        batch_enricher = BatchEnricher(pipeline, max_concurrent=2)
        results = await batch_enricher.enrich_batch(orgs, save_checkpoint_every=2)

        # Verify
        assert len(results) == 3
        for r in results:
            assert "organization" in r
            assert "enrichment" in r
            assert r["enrichment"]["success"] is True

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_budget_exceeded():
    """Test that enrichment stops when budget is exceeded."""
    with patch('nonprofit_enrichment_system.cost_tracker') as mock_tracker:
        mock_tracker.can_make_request.return_value = False

        pipeline = SimpleEnrichmentPipeline()
        result = await pipeline.enrich_organization(
            name="Test Org",
            address="Test Address",
            skip_cache=True
        )

        assert result.success is False
        assert any("budget" in err.lower() for err in result.errors)

@pytest.mark.asyncio
async def test_invalid_address():
    """Test handling of invalid/not found address."""
    with patch('nonprofit_enrichment_system.get_http_session') as mock_session:
        # Mock empty geocoding response
        mock_geo_response = AsyncMock()
        mock_geo_response.status = 200
        mock_geo_response.json = AsyncMock(return_value=[])  # No results
        mock_geo_response.raise_for_status = MagicMock()

        mock_session_instance = AsyncMock()
        mock_session_instance.get = AsyncMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_geo_response)
        ))
        mock_session.return_value = mock_session_instance

        result = await demographics_tool("Invalid Address 12345")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_parallel_execution():
    """Test that tools run in parallel for performance."""
    import time

    with patch('nonprofit_enrichment_system.demographics_tool') as mock_demo, \
         patch('nonprofit_enrichment_system.contact_social_mission_tool') as mock_contact, \
         patch('nonprofit_enrichment_system.news_tool') as mock_news, \
         patch('nonprofit_enrichment_system.leadership_tool') as mock_leadership:

        # Each mock sleeps for 1 second
        async def slow_mock(*args, **kwargs):
            await asyncio.sleep(1)
            return {"success": True}

        mock_demo.side_effect = slow_mock
        mock_contact.side_effect = slow_mock
        mock_news.side_effect = slow_mock
        mock_leadership.side_effect = slow_mock

        # Run enrichment
        pipeline = SimpleEnrichmentPipeline()
        start = time.time()
        result = await pipeline.enrich_organization(
            name="Test Org",
            address="Test Address",
            skip_cache=True
        )
        elapsed = time.time() - start

        # Should take ~1 second (parallel) not ~4 seconds (sequential)
        assert elapsed < 2.0  # Allow some overhead

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
