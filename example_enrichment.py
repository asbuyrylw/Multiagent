"""
Example Script: Nonprofit Enrichment

This script demonstrates basic usage of the nonprofit enrichment system.

Usage:
    python example_enrichment.py
"""

import asyncio
import json
from nonprofit_enrichment_system import (
    SimpleEnrichmentPipeline,
    BatchEnricher,
    cost_tracker,
    close_http_session
)


async def example_single_organization():
    """Example 1: Enrich a single organization."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Organization Enrichment")
    print("="*80 + "\n")

    pipeline = SimpleEnrichmentPipeline()

    # Enrich a well-known nonprofit
    result = await pipeline.enrich_organization(
        name="American Red Cross",
        address="430 17th St NW, Washington, DC 20006"
    )

    # Display results
    print(f"Organization: American Red Cross")
    print(f"Success: {result.success}")
    print(f"Quality Score: {result.metadata.get('quality_score', 0)}%")
    print(f"Quality Tier: {result.metadata.get('quality_tier', 'unknown')}")
    print(f"Processing Time: {result.metadata.get('processing_time_seconds', 0)}s")

    print("\n" + "-"*80)
    print("DEMOGRAPHICS")
    print("-"*80)
    if result.demographics.get('success'):
        print(f"Population: {result.demographics.get('population', 'N/A')}")
        print(f"Median Income: {result.demographics.get('median_income', 'N/A')}")
        print(f"Source: {result.demographics.get('source', 'N/A')}")
    else:
        print(f"Error: {result.demographics.get('error', 'Unknown error')}")

    print("\n" + "-"*80)
    print("CONTACT & MISSION")
    print("-"*80)
    if result.contact_social_mission.get('success'):
        print(f"Contact: {result.contact_social_mission.get('contact', 'N/A')}")
        print(f"Website: {result.contact_social_mission.get('website', 'N/A')}")
        print(f"Social Media: {len(result.contact_social_mission.get('social', []))} links")
        mission = result.contact_social_mission.get('mission', 'N/A')
        if len(mission) > 100:
            mission = mission[:97] + "..."
        print(f"Mission: {mission}")
    else:
        print(f"Error: {result.contact_social_mission.get('error', 'Unknown error')}")

    print("\n" + "-"*80)
    print("NEWS")
    print("-"*80)
    if result.news:
        for i, news_item in enumerate(result.news[:3], 1):
            print(f"{i}. {news_item}")
    else:
        print("No recent news found")

    print("\n" + "-"*80)
    print("LEADERSHIP")
    print("-"*80)
    if result.leadership.get('success'):
        leaders = result.leadership.get('leadership', {})
        print(f"Executive Director: {leaders.get('executive_director', 'N/A')}")
        print(f"Development Director: {leaders.get('development_director', 'N/A')}")
        print(f"Other Leaders: {leaders.get('other_leaders', 'N/A')}")
    else:
        print(f"Error: {result.leadership.get('error', 'Unknown error')}")

    # Show errors if any
    if result.errors:
        print("\n" + "-"*80)
        print("ERRORS")
        print("-"*80)
        for error in result.errors:
            print(f"• {error}")

    return result


async def example_batch_processing():
    """Example 2: Batch process multiple organizations."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Processing")
    print("="*80 + "\n")

    # Sample organizations
    orgs = [
        {
            "name": "Feeding America",
            "address": "35 E Wacker Dr, Chicago, IL 60601"
        },
        {
            "name": "United Way",
            "address": "1800 Diagonal Rd, Alexandria, VA 22314"
        },
        {
            "name": "Habitat for Humanity",
            "address": "285 Peachtree Center Ave NE, Atlanta, GA 30303"
        }
    ]

    print(f"Processing {len(orgs)} organizations...")
    print()

    # Set up batch enrichment with max 2 concurrent requests
    pipeline = SimpleEnrichmentPipeline()
    batch_enricher = BatchEnricher(pipeline, max_concurrent=2)

    # Process batch
    results = await batch_enricher.enrich_batch(
        orgs,
        save_checkpoint_every=2  # Save checkpoint every 2 orgs
    )

    # Analyze results
    print("\n" + "-"*80)
    print("BATCH RESULTS")
    print("-"*80 + "\n")

    total = len(results)
    successful = sum(1 for r in results if r["enrichment"]["success"])
    avg_quality = sum(
        r["enrichment"].get("metadata", {}).get("quality_score", 0)
        for r in results
    ) / total

    print(f"Total Processed: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Average Quality Score: {avg_quality:.1f}%")

    print("\n" + "-"*80)
    print("INDIVIDUAL RESULTS")
    print("-"*80 + "\n")

    for i, r in enumerate(results, 1):
        org = r["organization"]
        enrich = r["enrichment"]
        quality = enrich.get("metadata", {}).get("quality_score", 0)
        tier = enrich.get("metadata", {}).get("quality_tier", "unknown")

        print(f"{i}. {org['name']}")
        print(f"   Success: {enrich['success']}")
        print(f"   Quality: {quality}% ({tier})")
        print(f"   Errors: {len(enrich.get('errors', []))}")
        print()

    # Save full results to JSON
    output_file = "batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {output_file}")

    return results


async def example_cost_tracking():
    """Example 3: Cost tracking and budget management."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Cost Tracking")
    print("="*80 + "\n")

    # Get current cost stats
    stats = cost_tracker.get_stats()

    print("Current Cost Statistics:")
    print(f"  Today's Cost: ${stats['today_cost']:.4f}")
    print(f"  Daily Budget: ${stats['daily_budget']:.2f}")
    print(f"  Remaining Budget: ${stats['remaining_budget']:.2f}")
    print(f"  Total Requests Today: {stats['total_requests_today']}")
    print(f"  Cost Per Request: ${stats['cost_per_request']:.4f}")

    # Calculate projected capacity
    if stats['remaining_budget'] > 0 and stats['cost_per_request'] > 0:
        remaining_orgs = int(stats['remaining_budget'] / stats['cost_per_request'])
        print(f"\nProjected Capacity:")
        print(f"  Remaining Organizations: ~{remaining_orgs}")
    else:
        print(f"\nNote: Run some enrichments first to calculate cost per request")


async def main():
    """Main entry point - runs all examples."""
    try:
        # Example 1: Single organization
        await example_single_organization()

        # Example 2: Batch processing
        # Uncomment to run batch example:
        # await example_batch_processing()

        # Example 3: Cost tracking
        await example_cost_tracking()

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. View the monitoring dashboard:")
        print("   python nonprofit_dashboard.py")
        print("\n2. Run the test suite:")
        print("   pytest test_nonprofit_enrichment.py -v")
        print("\n3. Read the full documentation:")
        print("   cat NONPROFIT_ENRICHMENT_README.md")
        print("\n" + "="*80 + "\n")

    finally:
        # Clean up HTTP session
        await close_http_session()


if __name__ == "__main__":
    asyncio.run(main())
