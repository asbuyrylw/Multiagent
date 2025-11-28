"""
Nonprofit Enrichment Monitoring Dashboard

A simple CLI dashboard to monitor enrichment quality, costs, and cache statistics.

Usage:
    python nonprofit_dashboard.py
    python nonprofit_dashboard.py --export report.json
    python nonprofit_dashboard.py --cleanup 30  # Clean cache older than 30 days
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from nonprofit_enrichment_system import OrgData, Base, cleanup_old_cache


class EnrichmentDashboard:
    """Dashboard for monitoring enrichment system."""

    def __init__(self, db_path: str = "nonprofit_cache.db"):
        """
        Initialize dashboard.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_cache_stats(self) -> Dict:
        """Get overall cache statistics."""
        total_orgs = self.session.query(OrgData).count()

        if total_orgs == 0:
            return {
                "total_organizations": 0,
                "average_quality_score": 0,
                "quality_distribution": {},
                "recent_additions": [],
                "cache_age_stats": {},
            }

        # Average quality score
        avg_quality = self.session.query(
            func.avg(OrgData.data_quality_score)
        ).scalar() or 0

        # Quality distribution
        high_quality = self.session.query(OrgData).filter(
            OrgData.data_quality_score >= 70
        ).count()

        medium_quality = self.session.query(OrgData).filter(
            OrgData.data_quality_score >= 40,
            OrgData.data_quality_score < 70
        ).count()

        low_quality = self.session.query(OrgData).filter(
            OrgData.data_quality_score < 40
        ).count()

        # Recent additions (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent = self.session.query(OrgData).filter(
            OrgData.created_at >= week_ago
        ).order_by(OrgData.created_at.desc()).limit(10).all()

        recent_list = [
            {
                "name": org.name,
                "quality_score": org.data_quality_score,
                "added": org.created_at.strftime("%Y-%m-%d %H:%M")
            }
            for org in recent
        ]

        # Cache age distribution
        now = datetime.utcnow()
        fresh = self.session.query(OrgData).filter(
            OrgData.last_updated >= now - timedelta(days=7)
        ).count()

        stale = self.session.query(OrgData).filter(
            OrgData.last_updated >= now - timedelta(days=30),
            OrgData.last_updated < now - timedelta(days=7)
        ).count()

        old = self.session.query(OrgData).filter(
            OrgData.last_updated < now - timedelta(days=30)
        ).count()

        return {
            "total_organizations": total_orgs,
            "average_quality_score": round(avg_quality, 1),
            "quality_distribution": {
                "high (70-100%)": high_quality,
                "medium (40-69%)": medium_quality,
                "low (0-39%)": low_quality
            },
            "recent_additions_last_7_days": len(recent_list),
            "recent_organizations": recent_list,
            "cache_freshness": {
                "fresh (<7 days)": fresh,
                "stale (7-30 days)": stale,
                "old (>30 days)": old
            }
        }

    def get_top_organizations(self, limit: int = 10) -> List[Dict]:
        """Get top organizations by quality score."""
        top_orgs = self.session.query(OrgData).filter(
            OrgData.data_quality_score is not None
        ).order_by(
            OrgData.data_quality_score.desc()
        ).limit(limit).all()

        return [
            {
                "name": org.name,
                "address": org.address,
                "quality_score": org.data_quality_score,
                "last_updated": org.last_updated.strftime("%Y-%m-%d"),
                "query_count": org.query_count
            }
            for org in top_orgs
        ]

    def get_problematic_organizations(self, limit: int = 10) -> List[Dict]:
        """Get organizations with low quality scores."""
        problematic = self.session.query(OrgData).filter(
            OrgData.data_quality_score is not None,
            OrgData.data_quality_score < 40
        ).order_by(
            OrgData.data_quality_score.asc()
        ).limit(limit).all()

        return [
            {
                "name": org.name,
                "address": org.address,
                "quality_score": org.data_quality_score,
                "last_updated": org.last_updated.strftime("%Y-%m-%d"),
                "needs_review": True
            }
            for org in problematic
        ]

    def get_most_queried(self, limit: int = 10) -> List[Dict]:
        """Get most frequently queried organizations."""
        most_queried = self.session.query(OrgData).order_by(
            OrgData.query_count.desc()
        ).limit(limit).all()

        return [
            {
                "name": org.name,
                "query_count": org.query_count,
                "quality_score": org.data_quality_score,
                "last_updated": org.last_updated.strftime("%Y-%m-%d")
            }
            for org in most_queried
        ]

    def search_organization(self, search_term: str) -> List[Dict]:
        """Search for organizations by name."""
        results = self.session.query(OrgData).filter(
            OrgData.name.like(f"%{search_term}%")
        ).limit(20).all()

        return [
            {
                "name": org.name,
                "address": org.address,
                "quality_score": org.data_quality_score,
                "last_updated": org.last_updated.strftime("%Y-%m-%d %H:%M"),
                "data": json.loads(org.data_json)
            }
            for org in results
        ]

    def generate_report(self) -> Dict:
        """Generate comprehensive dashboard report."""
        return {
            "report_generated": datetime.utcnow().isoformat(),
            "cache_statistics": self.get_cache_stats(),
            "top_10_organizations": self.get_top_organizations(10),
            "problematic_organizations": self.get_problematic_organizations(10),
            "most_queried": self.get_most_queried(10)
        }

    def print_dashboard(self):
        """Print dashboard to console."""
        stats = self.get_cache_stats()

        print("\n" + "="*80)
        print("NONPROFIT ENRICHMENT DASHBOARD")
        print("="*80)

        print(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        print("\n" + "-"*80)
        print("CACHE OVERVIEW")
        print("-"*80)
        print(f"Total Organizations: {stats['total_organizations']}")
        print(f"Average Quality Score: {stats['average_quality_score']}%")

        print("\nQuality Distribution:")
        for tier, count in stats['quality_distribution'].items():
            percentage = (count / stats['total_organizations'] * 100) if stats['total_organizations'] > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {tier:20s}: {count:4d} ({percentage:5.1f}%) {bar}")

        print("\nCache Freshness:")
        for age, count in stats['cache_freshness'].items():
            percentage = (count / stats['total_organizations'] * 100) if stats['total_organizations'] > 0 else 0
            print(f"  {age:20s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\nRecent Additions (Last 7 Days): {stats['recent_additions_last_7_days']}")

        if stats['recent_organizations']:
            print("\nRecently Added Organizations:")
            for org in stats['recent_organizations'][:5]:
                print(f"  • {org['name'][:50]:50s} Quality: {org['quality_score']:3d}% ({org['added']})")

        # Top organizations
        print("\n" + "-"*80)
        print("TOP 10 ORGANIZATIONS (BY QUALITY)")
        print("-"*80)
        top_orgs = self.get_top_organizations(10)
        for i, org in enumerate(top_orgs, 1):
            print(f"{i:2d}. {org['name'][:50]:50s} Score: {org['quality_score']:3d}% | Queries: {org['query_count']:3d}")

        # Problematic organizations
        print("\n" + "-"*80)
        print("ORGANIZATIONS NEEDING REVIEW (LOW QUALITY)")
        print("-"*80)
        problematic = self.get_problematic_organizations(10)
        if problematic:
            for i, org in enumerate(problematic, 1):
                print(f"{i:2d}. {org['name'][:50]:50s} Score: {org['quality_score']:3d}%")
        else:
            print("No low-quality organizations found!")

        # Most queried
        print("\n" + "-"*80)
        print("MOST FREQUENTLY QUERIED ORGANIZATIONS")
        print("-"*80)
        most_queried = self.get_most_queried(10)
        for i, org in enumerate(most_queried, 1):
            print(f"{i:2d}. {org['name'][:50]:50s} Queries: {org['query_count']:4d} | Quality: {org['quality_score']:3d}%")

        print("\n" + "="*80 + "\n")

    def export_report(self, filename: str):
        """Export dashboard report to JSON file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report exported to: {filename}")

    def interactive_search(self):
        """Interactive search mode."""
        print("\n" + "="*80)
        print("INTERACTIVE SEARCH MODE")
        print("="*80)
        print("Search for organizations by name. Type 'exit' to quit.\n")

        while True:
            search_term = input("Search: ").strip()

            if search_term.lower() in ['exit', 'quit', 'q']:
                break

            if not search_term:
                continue

            results = self.search_organization(search_term)

            if not results:
                print(f"No organizations found matching '{search_term}'")
                continue

            print(f"\nFound {len(results)} result(s):")
            for i, org in enumerate(results, 1):
                print(f"\n{i}. {org['name']}")
                print(f"   Address: {org['address']}")
                print(f"   Quality: {org['quality_score']}%")
                print(f"   Last Updated: {org['last_updated']}")

            # Show details for first result
            if results:
                show_details = input("\nShow full details for result #1? (y/n): ").strip().lower()
                if show_details == 'y':
                    print("\nFull Data:")
                    print(json.dumps(results[0]['data'], indent=2))

    def close(self):
        """Close database connection."""
        self.session.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nonprofit Enrichment Dashboard")
    parser.add_argument(
        '--db',
        default='nonprofit_cache.db',
        help='Path to database file (default: nonprofit_cache.db)'
    )
    parser.add_argument(
        '--export',
        metavar='FILE',
        help='Export report to JSON file'
    )
    parser.add_argument(
        '--search',
        action='store_true',
        help='Interactive search mode'
    )
    parser.add_argument(
        '--cleanup',
        type=int,
        metavar='DAYS',
        help='Clean up cache entries older than DAYS'
    )

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db):
        print(f"\n⚠ Database not found: {args.db}")
        print("Run the enrichment system first to create the database.\n")
        return

    dashboard = EnrichmentDashboard(args.db)

    try:
        # Cleanup mode
        if args.cleanup:
            print(f"\n🗑  Cleaning up cache entries older than {args.cleanup} days...")
            deleted = cleanup_old_cache(args.cleanup)
            print(f"✓ Deleted {deleted} old entries\n")

        # Export mode
        if args.export:
            dashboard.export_report(args.export)

        # Search mode
        elif args.search:
            dashboard.interactive_search()

        # Default: Show dashboard
        else:
            dashboard.print_dashboard()

            # Offer interactive options
            print("\nOptions:")
            print("  1. Search for organization")
            print("  2. Export report to JSON")
            print("  3. Exit")

            choice = input("\nSelect option (1-3): ").strip()

            if choice == '1':
                dashboard.interactive_search()
            elif choice == '2':
                filename = input("Export filename (default: report.json): ").strip()
                if not filename:
                    filename = "report.json"
                dashboard.export_report(filename)

    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
