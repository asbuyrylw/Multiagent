#!/usr/bin/env python3
"""
Comprehensive Redfin Cincinnati Data Scraper
Extracts homes for sale and past sales data for ML model training
"""

import requests
import json
import csv
import time
import re
import random
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
import pandas as pd
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedfinCincinnatiScraper:
    def __init__(self):
        self.base_url = "https://www.redfin.com"
        self.search_api_base = "https://www.redfin.com/stingray/api/gis"
        self.property_api_base = "https://www.redfin.com/stingray/api/home/details"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Cincinnati area coordinates and region info
        self.cincinnati_region = {
            'region_id': 3879,  # Cincinnati city region ID
            'region_type': 6,    # City type
            'bounds': {
                'north': 39.262,
                'south': 39.027,
                'east': -84.251,
                'west': -84.820
            }
        }
        
        self.delay_range = (1, 3)  # Random delay between requests
        
    def get_search_params(self, status='for_sale', num_homes=350, start=0):
        """Generate search parameters for Cincinnati area"""
        
        # Create polygon bounds for Cincinnati
        bounds = self.cincinnati_region['bounds']
        polygon = f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['south']},{bounds['east']},{bounds['north']},{bounds['west']},{bounds['north']},{bounds['west']},{bounds['south']}"
        
        params = {
            'al': 1,
            'region_id': self.cincinnati_region['region_id'],
            'region_type': self.cincinnati_region['region_type'],
            'num_homes': num_homes,
            'start': start,
            'poly': polygon,
            'sf': '1,2,3,5,6,7',  # Property types
            'uipt': '1,2,3,4,5,6,7,8',  # UI property types
            'v': 8,
            'market': 'cincinnati'
        }
        
        if status == 'for_sale':
            params.update({
                'status': 9,  # Active listings
                'ord': 'redfin-recommended-asc'
            })
        elif status == 'sold':
            params.update({
                'status': 2,  # Sold
                'ord': 'sold-date-desc',
                'sold_within_days': 365  # Last year of sales
            })
        elif status == 'pending':
            params.update({
                'status': 10,  # Pending
                'ord': 'redfin-recommended-asc'
            })
        
        return params
    
    def search_properties(self, status='for_sale', max_results=1000):
        """Search for properties in Cincinnati"""
        logger.info(f"Searching for {status} properties in Cincinnati...")
        
        all_properties = []
        start = 0
        batch_size = 350
        
        while start < max_results:
            try:
                params = self.get_search_params(status=status, num_homes=batch_size, start=start)
                url = f"{self.search_api_base}?{urlencode(params)}"
                
                logger.info(f"Fetching batch starting at {start}...")
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for search request")
                    break
                
                data = response.json()
                
                if 'payload' not in data or 'homes' not in data['payload']:
                    logger.warning("No homes data in response")
                    break
                
                homes = data['payload']['homes']
                if not homes:
                    logger.info("No more properties found")
                    break
                
                logger.info(f"Found {len(homes)} properties in this batch")
                all_properties.extend(homes)
                
                start += batch_size
                time.sleep(random.uniform(*self.delay_range))
                
            except Exception as e:
                logger.error(f"Error in search: {e}")
                break
        
        logger.info(f"Total properties found: {len(all_properties)}")
        return all_properties
    
    def get_property_details(self, property_id, listing_id=None):
        """Get detailed property information"""
        try:
            url = f"{self.property_api_base}/propertyId/{property_id}"
            if listing_id:
                url += f"?listingId={listing_id}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for property {property_id}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting property details for {property_id}: {e}")
            return None
    
    def extract_property_features(self, home_data, detailed_data=None):
        """Extract ML-relevant features from property data"""
        features = {}
        
        # Basic property info
        features['property_id'] = home_data.get('propertyId')
        features['listing_id'] = home_data.get('listingId')
        features['mls_id'] = home_data.get('mlsId', {}).get('value') if home_data.get('mlsId') else None
        
        # Location
        features['address'] = home_data.get('streetLine', {}).get('value', '')
        features['city'] = home_data.get('city', '')
        features['state'] = home_data.get('state', '')
        features['zip_code'] = home_data.get('zip', '')
        features['latitude'] = home_data.get('latLong', {}).get('latitude')
        features['longitude'] = home_data.get('latLong', {}).get('longitude')
        
        # Property characteristics
        features['price'] = home_data.get('price', {}).get('value')
        features['beds'] = home_data.get('beds')
        features['baths'] = home_data.get('baths')
        features['sqft'] = home_data.get('sqFt', {}).get('value')
        features['lot_size'] = home_data.get('lotSize', {}).get('value')
        features['year_built'] = home_data.get('yearBuilt', {}).get('value')
        features['property_type'] = home_data.get('propertyType')
        features['home_type'] = home_data.get('homeType')
        
        # Financial info
        features['price_per_sqft'] = home_data.get('pricePerSqFt', {}).get('value')
        features['hoa_fee'] = home_data.get('hoa', {}).get('value')
        features['property_tax'] = home_data.get('propertyTax', {}).get('value')
        
        # Market info
        features['days_on_market'] = home_data.get('dom')
        features['status'] = home_data.get('homeStatus')
        features['listing_date'] = home_data.get('listingRemarks', {}).get('listDate')
        features['sold_date'] = home_data.get('soldDate')
        features['off_market_date'] = home_data.get('offMarketDate')
        
        # School and neighborhood
        features['elementary_school'] = home_data.get('schools', {}).get('elementary', {}).get('name')
        features['middle_school'] = home_data.get('schools', {}).get('middle', {}).get('name')
        features['high_school'] = home_data.get('schools', {}).get('high', {}).get('name')
        
        # Additional details from detailed API if available
        if detailed_data and 'payload' in detailed_data:
            payload = detailed_data['payload']
            
            # More detailed info
            features['garage_spaces'] = payload.get('publicRecordsInfo', {}).get('garageSpaces')
            features['parking_spaces'] = payload.get('publicRecordsInfo', {}).get('parkingSpaces')
            features['stories'] = payload.get('publicRecordsInfo', {}).get('stories')
            features['pool'] = bool(payload.get('amenitiesInfo', {}).get('hasPool'))
            features['fireplace'] = bool(payload.get('amenitiesInfo', {}).get('hasFireplace'))
            features['ac'] = bool(payload.get('amenitiesInfo', {}).get('hasAC'))
            
            # Price history
            price_history = payload.get('propertyHistoryInfo', {}).get('events', [])
            if price_history:
                features['price_history'] = [
                    {
                        'date': event.get('eventDate'),
                        'price': event.get('price'),
                        'event_type': event.get('eventDescription')
                    }
                    for event in price_history[:5]  # Last 5 events
                ]
        
        # Market insights
        features['redfin_estimate'] = home_data.get('redfin_estimate', {}).get('value')
        features['market_insights'] = {
            'price_insights': home_data.get('insights', {}).get('phrases', []),
            'competitive_insights': home_data.get('competitiveInsights', [])
        }
        
        # Listing agent info
        features['listing_agent'] = home_data.get('listingAgent', {}).get('name')
        features['listing_office'] = home_data.get('listingOffice', {}).get('name')
        
        # Data collection timestamp
        features['scraped_at'] = datetime.now().isoformat()
        
        return features
    
    def scrape_for_sale_properties(self, max_results=1000, get_details=True):
        """Scrape current for-sale properties"""
        logger.info("Starting to scrape for-sale properties...")
        
        properties = self.search_properties(status='for_sale', max_results=max_results)
        detailed_properties = []
        
        for i, home in enumerate(properties, 1):
            try:
                logger.info(f"Processing property {i}/{len(properties)}: {home.get('streetLine', {}).get('value', 'Unknown')}")
                
                features = self.extract_property_features(home)
                
                # Get detailed info if requested
                if get_details and home.get('propertyId'):
                    detailed_data = self.get_property_details(
                        home.get('propertyId'),
                        home.get('listingId')
                    )
                    if detailed_data:
                        features = self.extract_property_features(home, detailed_data)
                
                detailed_properties.append(features)
                
                # Rate limiting
                if i % 10 == 0:
                    time.sleep(random.uniform(2, 4))
                else:
                    time.sleep(random.uniform(*self.delay_range))
                    
            except Exception as e:
                logger.error(f"Error processing property {i}: {e}")
                continue
        
        return detailed_properties
    
    def scrape_sold_properties(self, max_results=2000, get_details=False):
        """Scrape recently sold properties for training data"""
        logger.info("Starting to scrape sold properties...")
        
        properties = self.search_properties(status='sold', max_results=max_results)
        detailed_properties = []
        
        for i, home in enumerate(properties, 1):
            try:
                logger.info(f"Processing sold property {i}/{len(properties)}: {home.get('streetLine', {}).get('value', 'Unknown')}")
                
                features = self.extract_property_features(home)
                
                # Get detailed info if requested
                if get_details and home.get('propertyId'):
                    detailed_data = self.get_property_details(
                        home.get('propertyId'),
                        home.get('listingId')
                    )
                    if detailed_data:
                        features = self.extract_property_features(home, detailed_data)
                
                detailed_properties.append(features)
                
                # Rate limiting for sold properties (less aggressive since it's historical data)
                if i % 20 == 0:
                    time.sleep(random.uniform(1, 2))
                else:
                    time.sleep(random.uniform(0.5, 1))
                    
            except Exception as e:
                logger.error(f"Error processing sold property {i}: {e}")
                continue
        
        return detailed_properties
    
    def save_data(self, data, filename, format='csv'):
        """Save scraped data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filename = f"{filename}_{timestamp}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(data)} records to {filename}")
            
        elif format == 'json':
            filename = f"{filename}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(data)} records to {filename}")
        
        return filename
    
    def create_ml_dataset(self, for_sale_data, sold_data):
        """Create a combined dataset optimized for ML training"""
        logger.info("Creating ML-optimized dataset...")
        
        # Combine datasets
        all_data = []
        
        # Mark data types
        for prop in for_sale_data:
            prop['data_type'] = 'for_sale'
            prop['target_sold'] = 0  # Binary target for ML
            all_data.append(prop)
        
        for prop in sold_data:
            prop['data_type'] = 'sold'
            prop['target_sold'] = 1  # Binary target for ML
            all_data.append(prop)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(all_data)
        
        # Add derived features for ML
        df['price_per_bed'] = df['price'] / df['beds'].replace(0, 1)
        df['price_per_bath'] = df['price'] / df['baths'].replace(0, 1)
        df['bed_bath_ratio'] = df['beds'] / df['baths'].replace(0, 1)
        df['sqft_per_bed'] = df['sqft'] / df['beds'].replace(0, 1)
        df['age_of_home'] = 2024 - pd.to_numeric(df['year_built'], errors='coerce')
        
        # Calculate days since listing for active properties
        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            df['days_since_listing'] = (datetime.now() - df['listing_date']).dt.days
        
        # Clean and prepare numeric columns
        numeric_columns = ['price', 'beds', 'baths', 'sqft', 'lot_size', 'year_built', 
                          'price_per_sqft', 'hoa_fee', 'property_tax', 'days_on_market']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add market segment features
        df['price_segment'] = pd.cut(df['price'], 
                                   bins=[0, 150000, 300000, 500000, 750000, float('inf')], 
                                   labels=['Budget', 'Moderate', 'Upper_Mid', 'Luxury', 'Ultra_Luxury'])
        
        df['size_segment'] = pd.cut(df['sqft'], 
                                  bins=[0, 1200, 1800, 2500, 3500, float('inf')], 
                                  labels=['Small', 'Medium', 'Large', 'Very_Large', 'Mansion'])
        
        return df
    
    def generate_summary_report(self, df):
        """Generate summary statistics and insights"""
        logger.info("Generating summary report...")
        
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_properties': len(df),
            'for_sale_count': len(df[df['data_type'] == 'for_sale']),
            'sold_count': len(df[df['data_type'] == 'sold']),
            'price_statistics': {
                'mean': df['price'].mean(),
                'median': df['price'].median(),
                'min': df['price'].min(),
                'max': df['price'].max(),
                'std': df['price'].std()
            },
            'property_types': df['property_type'].value_counts().to_dict(),
            'price_segments': df['price_segment'].value_counts().to_dict(),
            'size_segments': df['size_segment'].value_counts().to_dict(),
            'beds_distribution': df['beds'].value_counts().to_dict(),
            'baths_distribution': df['baths'].value_counts().to_dict(),
            'geographic_distribution': df['zip_code'].value_counts().head(10).to_dict()
        }
        
        return report

def main():
    """Main execution function"""
    scraper = RedfinCincinnatiScraper()
    
    print("=" * 60)
    print("REDFIN CINCINNATI DATA SCRAPER")
    print("Extracting property data for ML model training")
    print("=" * 60)
    
    try:
        # Scrape current for-sale properties
        print("\n1. Scraping current for-sale properties...")
        for_sale_data = scraper.scrape_for_sale_properties(max_results=500, get_details=True)
        for_sale_file = scraper.save_data(for_sale_data, 'cincinnati_for_sale', 'csv')
        
        # Scrape recently sold properties
        print("\n2. Scraping recently sold properties...")
        sold_data = scraper.scrape_sold_properties(max_results=1000, get_details=False)
        sold_file = scraper.save_data(sold_data, 'cincinnati_sold', 'csv')
        
        # Create combined ML dataset
        print("\n3. Creating ML-optimized dataset...")
        ml_dataset = scraper.create_ml_dataset(for_sale_data, sold_data)
        ml_file = scraper.save_data(ml_dataset.to_dict('records'), 'cincinnati_ml_dataset', 'csv')
        
        # Generate summary report
        print("\n4. Generating summary report...")
        summary = scraper.generate_summary_report(ml_dataset)
        summary_file = scraper.save_data([summary], 'cincinnati_market_summary', 'json')
        
        print("\n" + "=" * 60)
        print("DATA EXTRACTION COMPLETE")
        print("=" * 60)
        
        print(f"\nFiles created:")
        print(f"✓ For-sale properties: {for_sale_file}")
        print(f"✓ Sold properties: {sold_file}")
        print(f"✓ ML dataset: {ml_file}")
        print(f"✓ Market summary: {summary_file}")
        
        print(f"\nDataset Summary:")
        print(f"• Total properties: {len(ml_dataset)}")
        print(f"• For sale: {summary['for_sale_count']}")
        print(f"• Recently sold: {summary['sold_count']}")
        print(f"• Price range: ${summary['price_statistics']['min']:,.0f} - ${summary['price_statistics']['max']:,.0f}")
        print(f"• Median price: ${summary['price_statistics']['median']:,.0f}")
        
        print(f"\nTop property types:")
        for prop_type, count in list(summary['property_types'].items())[:5]:
            print(f"• {prop_type}: {count}")
        
        print("\nReady for ML model training! 🚀")
        
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()