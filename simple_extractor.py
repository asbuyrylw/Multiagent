#!/usr/bin/env python3
"""
Simple Tableau Data Extractor using only standard library
"""

import urllib.request
import urllib.parse
import json
import re
import time
import ssl
from html.parser import HTMLParser

class SimpleTableauExtractor:
    def __init__(self, viz_url):
        self.viz_url = viz_url
        # Create SSL context that doesn't verify certificates
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Set up headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_page(self, url):
        """Fetch a web page and return content"""
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=30, context=self.ssl_context) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_workbook_name(self):
        """Extract workbook name from Tableau URL"""
        parts = self.viz_url.split('/')
        if 'viz' in parts:
            viz_index = parts.index('viz')
            if len(parts) > viz_index + 1:
                return parts[viz_index + 1]
        return None
    
    def try_csv_download(self):
        """Try to download CSV directly"""
        print("=== Trying Direct CSV Download ===")
        
        workbook_name = self.extract_workbook_name()
        if not workbook_name:
            print("Could not extract workbook name")
            return None
        
        # Common CSV URL patterns for Tableau Public
        csv_urls = [
            f"https://public.tableau.com/views/{workbook_name}/DS1.csv",
            f"https://public.tableau.com/workbooks/{workbook_name}.csv",
            f"{self.viz_url}.csv",
            f"{self.viz_url}?:download=yes&:format=csv"
        ]
        
        for csv_url in csv_urls:
            print(f"Trying: {csv_url}")
            try:
                req = urllib.request.Request(csv_url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=30, context=self.ssl_context) as response:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'csv' in content_type or response.getcode() == 200:
                        content = response.read().decode('utf-8')
                        if content.startswith('Date,') or ',' in content[:100]:  # Basic CSV check
                            print("✓ Found CSV data!")
                            return content
                        else:
                            print("Response doesn't look like CSV data")
            except Exception as e:
                print(f"Error: {e}")
        
        print("No direct CSV access available")
        return None
    
    def scrape_dashboard(self):
        """Scrape the dashboard page for data"""
        print("\n=== Scraping Dashboard Page ===")
        
        content = self.fetch_page(self.viz_url)
        if not content:
            return None
        
        # Look for JSON data patterns
        json_patterns = [
            r'"dataSource":\s*(\{[^}]+\})',
            r'"data":\s*(\[[^\]]+\])',
            r'"worksheetData":\s*(\{[^}]+\})',
            r'var\s+data\s*=\s*(\{[^}]+\});',
            r'data:\s*(\[[^\]]+\])'
        ]
        
        found_data = []
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, (dict, list)) and data:
                        found_data.append(data)
                        print(f"✓ Found JSON data structure with {len(data) if isinstance(data, (list, dict)) else 0} items")
                except json.JSONDecodeError:
                    continue
        
        # Look for CSV-like data in the page
        csv_patterns = [
            r'([A-Za-z]+(?:,\s*[A-Za-z]+)+\n(?:[^,\n]+(?:,\s*[^,\n]+)+\n?)+)',
            r'Date,.*\n(?:\d{4}-\d{2}-\d{2},.*\n?)+'
        ]
        
        for pattern in csv_patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"✓ Found potential CSV data patterns: {len(matches)}")
                found_data.extend(matches)
        
        return found_data if found_data else None
    
    def check_fhfa_sources(self):
        """Check official FHFA data sources"""
        print("\n=== Checking Official FHFA Sources ===")
        
        fhfa_urls = [
            "https://www.fhfa.gov/data/national-mortgage-database-aggregate-statistics",
            "https://www.fhfa.gov/document/nmdb-outstanding-mortgage-statistics-all-quarterly.zip",
            "https://www.fhfa.gov/document/nmdb-new-mortgage-statistics-all-annual.zip"
        ]
        
        available_sources = []
        for url in fhfa_urls:
            try:
                req = urllib.request.Request(url, headers=self.headers)
                req.get_method = lambda: 'HEAD'  # Use HEAD request to check availability
                with urllib.request.urlopen(req, timeout=15, context=self.ssl_context) as response:
                    if response.getcode() == 200:
                        print(f"✓ Available: {url}")
                        available_sources.append(url)
                    else:
                        print(f"✗ Not available ({response.getcode()}): {url}")
            except Exception as e:
                print(f"✗ Error checking {url}: {e}")
        
        return available_sources
    
    def download_zip_file(self, url, filename=None):
        """Download a ZIP file from URL"""
        if not filename:
            filename = url.split('/')[-1]
        
        print(f"\nDownloading: {url}")
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=60, context=self.ssl_context) as response:
                if response.getcode() == 200:
                    with open(filename, 'wb') as f:
                        f.write(response.read())
                    print(f"✓ Downloaded: {filename}")
                    return filename
                else:
                    print(f"Download failed: {response.getcode()}")
                    return None
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def extract_all(self):
        """Run all extraction methods"""
        print("NMDB Tableau Data Extraction")
        print(f"Target: {self.viz_url}")
        print("=" * 50)
        
        results = {}
        
        # Method 1: Try direct CSV
        results['csv_data'] = self.try_csv_download()
        
        # Method 2: Scrape page
        results['scraped_data'] = self.scrape_dashboard()
        
        # Method 3: Check official sources
        results['official_sources'] = self.check_fhfa_sources()
        
        # Download available ZIP files
        if results['official_sources']:
            print("\n=== Downloading Official Data ===")
            downloaded_files = []
            for source in results['official_sources']:
                if source.endswith('.zip'):
                    filename = self.download_zip_file(source)
                    if filename:
                        downloaded_files.append(filename)
            results['downloaded_files'] = downloaded_files
        
        return results
    
    def save_results(self, results):
        """Save extracted results"""
        print("\n=== Saving Results ===")
        
        # Save CSV data if found
        if results.get('csv_data'):
            with open('tableau_data.csv', 'w') as f:
                f.write(results['csv_data'])
            print("✓ Saved CSV data to tableau_data.csv")
        
        # Save scraped data
        if results.get('scraped_data'):
            with open('scraped_data.json', 'w') as f:
                json.dump(results['scraped_data'], f, indent=2)
            print("✓ Saved scraped data to scraped_data.json")
        
        # Save summary
        summary = {
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_url': self.viz_url,
            'csv_found': bool(results.get('csv_data')),
            'scraped_data_found': bool(results.get('scraped_data')),
            'official_sources': results.get('official_sources', []),
            'downloaded_files': results.get('downloaded_files', [])
        }
        
        with open('extraction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("✓ Saved summary to extraction_summary.json")
        
        return summary

def main():
    tableau_url = "https://public.tableau.com/app/profile/nmdb.fhfa/viz/NMDBDashboardModel1v1/DS1"
    
    extractor = SimpleTableauExtractor(tableau_url)
    results = extractor.extract_all()
    summary = extractor.save_results(results)
    
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    
    if summary['csv_found']:
        print("✓ CSV data extracted from Tableau")
    else:
        print("✗ No CSV data found from Tableau")
    
    if summary['scraped_data_found']:
        print("✓ Scraped data from dashboard page")
    else:
        print("✗ No structured data found on page")
    
    if summary['official_sources']:
        print(f"✓ Found {len(summary['official_sources'])} official FHFA sources")
        for source in summary['official_sources']:
            print(f"  - {source}")
    
    if summary['downloaded_files']:
        print(f"✓ Downloaded {len(summary['downloaded_files'])} files:")
        for file in summary['downloaded_files']:
            print(f"  - {file}")
    
    print("\nRECOMMENDATION:")
    print("The FHFA provides official CSV/ZIP downloads that contain")
    print("the same data as the Tableau dashboard, often in more detail.")
    print("These official sources are preferred over scraping Tableau.")

if __name__ == "__main__":
    main()