#!/usr/bin/env python3
"""
Tableau Data Extractor for NMDB FHFA Dashboard
Extracts data from: https://public.tableau.com/app/profile/nmdb.fhfa/viz/NMDBDashboardModel1v1/DS1
"""

import requests
import pandas as pd
import json
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import sys
import os

class TableauDataExtractor:
    def __init__(self, viz_url):
        self.viz_url = viz_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def extract_workbook_name(self):
        """Extract workbook name from Tableau URL"""
        # URL format: https://public.tableau.com/app/profile/username/viz/WorkbookName/ViewName
        parts = self.viz_url.split('/')
        if 'viz' in parts:
            viz_index = parts.index('viz')
            if len(parts) > viz_index + 1:
                return parts[viz_index + 1]
        return None
    
    def method1_csv_download(self):
        """Method 1: Try direct CSV download by modifying URL"""
        print("=== Method 1: Direct CSV Download ===")
        
        workbook_name = self.extract_workbook_name()
        if not workbook_name:
            print("Could not extract workbook name from URL")
            return None
            
        # Try different CSV URL patterns
        csv_patterns = [
            f"https://public.tableau.com/vizql/v_202334.24.0815.1719/bootstrapSession/sessions/{{session_id}}",
            f"https://public.tableau.com/views/{workbook_name}/DS1.csv",
            f"https://public.tableau.com/workbooks/{workbook_name}.csv",
        ]
        
        for pattern in csv_patterns:
            try:
                print(f"Trying: {pattern}")
                response = self.session.get(pattern, timeout=30)
                if response.status_code == 200 and 'text/csv' in response.headers.get('content-type', ''):
                    print("✓ CSV data found!")
                    return response.text
                elif response.status_code == 200:
                    print(f"Response received but content-type: {response.headers.get('content-type')}")
            except Exception as e:
                print(f"Error: {e}")
        
        print("No direct CSV download available")
        return None
    
    def method2_scrape_data(self):
        """Method 2: Scrape the dashboard page for embedded data"""
        print("\n=== Method 2: Web Scraping ===")
        
        try:
            response = self.session.get(self.viz_url, timeout=30)
            if response.status_code != 200:
                print(f"Failed to load page: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for JSON data in script tags
            scripts = soup.find_all('script')
            data_found = []
            
            for script in scripts:
                if script.string and ('data' in script.string.lower() or 'json' in script.string.lower()):
                    # Look for JSON-like structures
                    text = script.string
                    json_matches = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', text)
                    for match in json_matches:
                        try:
                            data = json.loads(match)
                            if isinstance(data, dict) and len(data) > 3:  # Filter meaningful data
                                data_found.append(data)
                        except json.JSONDecodeError:
                            continue
            
            if data_found:
                print(f"✓ Found {len(data_found)} potential data objects")
                return data_found
            else:
                print("No structured data found in page")
                return None
                
        except Exception as e:
            print(f"Error scraping page: {e}")
            return None
    
    def method3_tableau_api(self):
        """Method 3: Try to use Tableau's internal API"""
        print("\n=== Method 3: Tableau Internal API ===")
        
        try:
            # First, get the main page to extract session information
            response = self.session.get(self.viz_url, timeout=30)
            if response.status_code != 200:
                print("Failed to load main page")
                return None
            
            # Look for Tableau-specific tokens or session IDs
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find data-* attributes that might contain API endpoints
            elements_with_data = soup.find_all(attrs=lambda x: x and any(key.startswith('data-') for key in x.keys()))
            
            api_endpoints = []
            for element in elements_with_data:
                for attr, value in element.attrs.items():
                    if attr.startswith('data-') and ('url' in attr.lower() or 'api' in attr.lower()):
                        api_endpoints.append(value)
            
            if api_endpoints:
                print(f"Found {len(api_endpoints)} potential API endpoints")
                for endpoint in api_endpoints[:3]:  # Try first 3
                    try:
                        if endpoint.startswith('/'):
                            endpoint = urljoin(self.viz_url, endpoint)
                        print(f"Trying endpoint: {endpoint}")
                        api_response = self.session.get(endpoint, timeout=15)
                        if api_response.status_code == 200:
                            print(f"✓ API endpoint accessible: {endpoint}")
                            return api_response.text
                    except Exception as e:
                        print(f"Error accessing {endpoint}: {e}")
            
            print("No accessible API endpoints found")
            return None
            
        except Exception as e:
            print(f"Error with API method: {e}")
            return None
    
    def method4_alternative_sources(self):
        """Method 4: Check for alternative data sources from FHFA"""
        print("\n=== Method 4: Alternative Data Sources ===")
        
        # Based on the web search, we know FHFA provides CSV downloads
        fhfa_data_urls = [
            "https://www.fhfa.gov/data/national-mortgage-database-aggregate-statistics",
            "https://www.fhfa.gov/document/nmdb-outstanding-mortgage-statistics-all-quarterly.zip",
            "https://www.fhfa.gov/document/nmdb-new-mortgage-statistics-all-annual.zip",
            "https://www.fhfa.gov/document/nmdb-mortgage-performance-statistics-all-quarterly.zip"
        ]
        
        print("Checking FHFA official data sources...")
        accessible_sources = []
        
        for url in fhfa_data_urls:
            try:
                response = self.session.head(url, timeout=15)
                if response.status_code == 200:
                    print(f"✓ Accessible: {url}")
                    accessible_sources.append(url)
                else:
                    print(f"✗ Not accessible ({response.status_code}): {url}")
            except Exception as e:
                print(f"✗ Error checking {url}: {e}")
        
        return accessible_sources
    
    def download_csv_data(self, url):
        """Download and parse CSV data from URL"""
        try:
            print(f"\nDownloading data from: {url}")
            response = self.session.get(url, timeout=60)
            
            if response.status_code == 200:
                # Check if it's a ZIP file
                if url.endswith('.zip') or 'zip' in response.headers.get('content-type', ''):
                    print("Downloaded ZIP file - you'll need to extract it manually")
                    filename = url.split('/')[-1]
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Saved as: {filename}")
                    return filename
                
                # Try to parse as CSV
                elif 'csv' in response.headers.get('content-type', '') or url.endswith('.csv'):
                    df = pd.read_csv(pd.StringIO(response.text))
                    print(f"✓ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                    return df
                
                else:
                    print(f"Unknown content type: {response.headers.get('content-type')}")
                    return response.text
            else:
                print(f"Failed to download: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    
    def extract_all_methods(self):
        """Run all extraction methods"""
        print("Starting data extraction from NMDB Tableau Dashboard")
        print(f"Target URL: {self.viz_url}")
        print("=" * 60)
        
        results = {}
        
        # Try Method 1: Direct CSV
        results['csv_direct'] = self.method1_csv_download()
        
        # Try Method 2: Web scraping
        results['web_scraping'] = self.method2_scrape_data()
        
        # Try Method 3: Tableau API
        results['tableau_api'] = self.method3_tableau_api()
        
        # Try Method 4: Alternative sources
        results['alternative_sources'] = self.method4_alternative_sources()
        
        # If we found alternative sources, try to download them
        if results['alternative_sources']:
            print("\n=== Downloading from Alternative Sources ===")
            results['downloaded_data'] = []
            for source_url in results['alternative_sources'][:2]:  # Download first 2
                data = self.download_csv_data(source_url)
                if data is not None:
                    results['downloaded_data'].append(data)
        
        return results
    
    def save_results(self, results):
        """Save results to files"""
        print("\n=== Saving Results ===")
        
        # Save any scraped data as JSON
        if results.get('web_scraping'):
            with open('scraped_data.json', 'w') as f:
                json.dump(results['web_scraping'], f, indent=2)
            print("✓ Saved scraped data to scraped_data.json")
        
        # Save CSV data
        for i, data in enumerate(results.get('downloaded_data', [])):
            if isinstance(data, pd.DataFrame):
                filename = f'nmdb_data_{i+1}.csv'
                data.to_csv(filename, index=False)
                print(f"✓ Saved dataset to {filename}")
                
                # Show preview
                print(f"\nPreview of {filename}:")
                print(data.head())
                print(f"Shape: {data.shape}")
        
        # Save summary
        summary = {
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_url': self.viz_url,
            'methods_tried': list(results.keys()),
            'successful_methods': [k for k, v in results.items() if v],
            'alternative_sources_found': results.get('alternative_sources', [])
        }
        
        with open('extraction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("✓ Saved extraction summary to extraction_summary.json")

def main():
    tableau_url = "https://public.tableau.com/app/profile/nmdb.fhfa/viz/NMDBDashboardModel1v1/DS1"
    
    extractor = TableauDataExtractor(tableau_url)
    results = extractor.extract_all_methods()
    extractor.save_results(results)
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    
    # Summary
    successful_methods = [k for k, v in results.items() if v]
    if successful_methods:
        print(f"✓ Successful methods: {', '.join(successful_methods)}")
    else:
        print("✗ No methods were successful")
    
    if results.get('alternative_sources'):
        print(f"\n📊 Found {len(results['alternative_sources'])} alternative FHFA data sources")
        print("These official sources likely contain the same or better data than the Tableau dashboard")
    
    print("\nRecommendation:")
    print("The FHFA provides official CSV downloads that are likely more comprehensive")
    print("than extracting from the Tableau dashboard. Consider using those sources directly.")

if __name__ == "__main__":
    main()