#!/usr/bin/env python3
"""
Working Redfin Cincinnati Data Scraper
Properly handles JSONP response format and extracts property data for ML training
"""

import requests
import json
import csv
import time
import random
from datetime import datetime
from urllib.parse import urlencode
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingRedfinScraper:
    def __init__(self):
        self.base_url = "https://www.redfin.com"
        self.api_url = "https://www.redfin.com/stingray/api/gis"
        self.csv_url = "https://www.redfin.com/stingray/api/gis-csv"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.redfin.com/city/3879/OH/Cincinnati'
        })
        
        # Cincinnati region configuration
        self.cincinnati_config = {
            'region_id': 3879,
            'region_type': 6,
            'market': 'cincinnati'
        }
        
    def parse_jsonp_response(self, response_text):
        """Parse JSONP-style response from Redfin API"""
        if response_text.startswith('{}&&'):
            # Remove JSONP wrapper
            json_text = response_text[4:]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return None
        elif response_text.startswith('{'):
            # Standard JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return None
        else:
            logger.error("Unknown response format")
            return None
    
    def get_properties_api(self, status='active', max_homes=1000):
        """Get properties using the JSON API"""
        logger.info(f"Fetching {status} properties from Cincinnati...")
        
        # Map status to Redfin status codes
        status_map = {
            'active': 9,
            'sold': 2,
            'pending': 10,
            'off_market': 5
        }
        
        all_properties = []
        start = 0
        batch_size = 350
        
        while start < max_homes:
            params = {
                'al': 1,
                'region_id': self.cincinnati_config['region_id'],
                'region_type': self.cincinnati_config['region_type'],
                'num_homes': min(batch_size, max_homes - start),
                'start': start,
                'sf': '1,2,3,5,6,7',  # Property types
                'uipt': '1,2,3,4,5,6,7,8',  # UI property types
                'status': status_map.get(status, 9),
                'v': 8
            }
            
            # Add sold-specific parameters
            if status == 'sold':
                params['sold_within_days'] = 365  # Last year
                params['ord'] = 'sold-date-desc'
            else:
                params['ord'] = 'redfin-recommended-asc'
            
            try:
                url = f"{self.api_url}?{urlencode(params)}"
                logger.info(f"Fetching batch {start}-{start + batch_size}...")
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for batch starting at {start}")
                    break
                
                # Parse JSONP response
                data = self.parse_jsonp_response(response.text)
                if not data:
                    logger.error("Failed to parse response")
                    break
                
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
                
                # Rate limiting
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                break
        
        logger.info(f"Total properties collected: {len(all_properties)}")
        return all_properties
    
    def get_properties_csv(self, status='active'):
        """Get properties using the CSV API (alternative method)"""
        logger.info(f"Fetching {status} properties via CSV API...")
        
        status_map = {
            'active': 9,
            'sold': 2,
            'pending': 10
        }
        
        params = {
            'al': 1,
            'region_id': self.cincinnati_config['region_id'],
            'region_type': self.cincinnati_config['region_type'],
            'num_homes': 1000,
            'start': 0,
            'status': status_map.get(status, 9),
            'v': 8
        }
        
        try:
            url = f"{self.csv_url}?{urlencode(params)}"
            response = self.session.get(url, timeout=60)
            
            if response.status_code == 200:
                # Parse CSV response
                csv_data = response.text
                lines = csv_data.strip().split('\n')
                
                if len(lines) > 2:  # Header + disclaimer + data
                    # Skip header and disclaimer
                    data_lines = [line for line in lines if not line.startswith('"In accordance')]
                    
                    # Create DataFrame
                    from io import StringIO
                    df = pd.read_csv(StringIO('\n'.join(data_lines)))
                    logger.info(f"CSV API returned {len(df)} properties")
                    return df
                else:
                    logger.warning("CSV response contains no data")
                    return None
            else:
                logger.error(f"CSV API failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error with CSV API: {e}")
            return None
    
    def extract_property_features(self, home_data):
        """Extract ML-relevant features from property data"""
        features = {}
        
        # Helper function to safely get nested values
        def safe_get(data, key, subkey=None):
            if isinstance(data, dict):
                if subkey:
                    return data.get(key, {}).get(subkey)
                else:
                    return data.get(key)
            return None
        
        # Basic identifiers
        features['property_id'] = safe_get(home_data, 'propertyId')
        features['listing_id'] = safe_get(home_data, 'listingId')
        features['mls_id'] = safe_get(home_data, 'mlsId', 'value')
        
        # Location
        features['address'] = safe_get(home_data, 'streetLine', 'value')
        features['city'] = safe_get(home_data, 'city')
        features['state'] = safe_get(home_data, 'state')
        features['zip_code'] = safe_get(home_data, 'zip')
        features['latitude'] = safe_get(home_data, 'latLong', 'latitude')
        features['longitude'] = safe_get(home_data, 'latLong', 'longitude')
        features['location'] = safe_get(home_data, 'location', 'value')
        
        # Property characteristics
        features['price'] = safe_get(home_data, 'price', 'value')
        features['beds'] = safe_get(home_data, 'beds')
        features['baths'] = safe_get(home_data, 'baths')
        features['full_baths'] = safe_get(home_data, 'fullBaths')
        features['sqft'] = safe_get(home_data, 'sqFt', 'value')
        features['lot_size'] = safe_get(home_data, 'lotSize', 'value')
        features['year_built'] = safe_get(home_data, 'yearBuilt', 'value')
        features['stories'] = safe_get(home_data, 'stories')
        
        # Property type and status
        features['property_type'] = safe_get(home_data, 'propertyType')
        features['ui_property_type'] = safe_get(home_data, 'uiPropertyType')
        features['listing_type'] = safe_get(home_data, 'listingType')
        features['mls_status'] = safe_get(home_data, 'mlsStatus')
        features['search_status'] = safe_get(home_data, 'searchStatus')
        
        # Financial information
        features['price_per_sqft'] = safe_get(home_data, 'pricePerSqFt', 'value')
        features['hoa_fee'] = safe_get(home_data, 'hoa', 'value')
        
        # Market timing
        features['days_on_market'] = safe_get(home_data, 'dom')
        features['time_on_redfin'] = safe_get(home_data, 'timeOnRedfin')
        features['sold_date'] = safe_get(home_data, 'soldDate')
        
        # Features and amenities
        features['is_hot'] = safe_get(home_data, 'isHot')
        features['has_virtual_tour'] = safe_get(home_data, 'hasVirtualTour')
        features['has_video_tour'] = safe_get(home_data, 'hasVideoTour')
        features['has_3d_tour'] = safe_get(home_data, 'has3DTour')
        features['is_new_construction'] = safe_get(home_data, 'isNewConstruction')
        
        # Listing information
        features['listing_broker'] = safe_get(home_data, 'listingBroker', 'name')
        features['data_source_id'] = safe_get(home_data, 'dataSourceId')
        features['market_id'] = safe_get(home_data, 'marketId')
        
        # Additional metadata
        features['has_insight'] = safe_get(home_data, 'hasInsight')
        features['url'] = safe_get(home_data, 'url')
        features['scraped_at'] = datetime.now().isoformat()
        
        return features
    
    def create_ml_dataset(self, properties_data, data_type='active'):
        """Convert property data to ML-ready dataset"""
        logger.info(f"Creating ML dataset from {len(properties_data)} {data_type} properties...")
        
        # Extract features from all properties
        features_list = []
        for prop in properties_data:
            features = self.extract_property_features(prop)
            features['data_type'] = data_type
            features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Clean and prepare data
        df = self.clean_data(df)
        
        # Add derived features for ML
        df = self.add_derived_features(df)
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        # Convert numeric columns
        numeric_columns = ['price', 'beds', 'baths', 'sqft', 'lot_size', 'year_built', 
                          'price_per_sqft', 'hoa_fee', 'days_on_market', 'stories']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        boolean_columns = ['is_hot', 'has_virtual_tour', 'has_video_tour', 'has_3d_tour', 
                          'is_new_construction', 'has_insight']
        
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Clean text columns
        text_columns = ['address', 'city', 'location', 'property_type', 'mls_status']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def add_derived_features(self, df):
        """Add derived features for ML training"""
        # Price-based features
        if 'price' in df.columns and 'beds' in df.columns:
            df['price_per_bed'] = df['price'] / df['beds'].replace(0, 1)
        
        if 'price' in df.columns and 'baths' in df.columns:
            df['price_per_bath'] = df['price'] / df['baths'].replace(0, 1)
        
        # Size-based features
        if 'sqft' in df.columns and 'beds' in df.columns:
            df['sqft_per_bed'] = df['sqft'] / df['beds'].replace(0, 1)
        
        if 'beds' in df.columns and 'baths' in df.columns:
            df['bed_bath_ratio'] = df['beds'] / df['baths'].replace(0, 1)
        
        # Age of home
        if 'year_built' in df.columns:
            df['home_age'] = 2024 - df['year_built']
        
        # Price segments
        if 'price' in df.columns:
            df['price_segment'] = pd.cut(df['price'], 
                                       bins=[0, 150000, 300000, 500000, 750000, float('inf')], 
                                       labels=['Budget', 'Moderate', 'Upper_Mid', 'Luxury', 'Ultra_Luxury'])
        
        # Size segments
        if 'sqft' in df.columns:
            df['size_segment'] = pd.cut(df['sqft'], 
                                      bins=[0, 1200, 1800, 2500, 3500, float('inf')], 
                                      labels=['Small', 'Medium', 'Large', 'Very_Large', 'Mansion'])
        
        return df
    
    def save_dataset(self, df, filename_base, format='csv'):
        """Save the dataset to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filename = f"{filename_base}_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format == 'json':
            filename = f"{filename_base}_{timestamp}.json"
            df.to_json(filename, orient='records', indent=2)
        
        logger.info(f"Saved {len(df)} records to {filename}")
        return filename
    
    def generate_summary_stats(self, df):
        """Generate summary statistics"""
        stats = {
            'total_properties': len(df),
            'data_collection_date': datetime.now().isoformat(),
            'price_stats': {
                'mean': float(df['price'].mean()) if 'price' in df.columns else None,
                'median': float(df['price'].median()) if 'price' in df.columns else None,
                'min': float(df['price'].min()) if 'price' in df.columns else None,
                'max': float(df['price'].max()) if 'price' in df.columns else None
            },
            'property_types': df['property_type'].value_counts().to_dict() if 'property_type' in df.columns else {},
            'bed_distribution': df['beds'].value_counts().to_dict() if 'beds' in df.columns else {},
            'price_segments': df['price_segment'].value_counts().to_dict() if 'price_segment' in df.columns else {}
        }
        return stats

def main():
    """Main execution function"""
    scraper = WorkingRedfinScraper()
    
    print("=" * 60)
    print("REDFIN CINCINNATI DATA SCRAPER")
    print("Extracting property data for ML model training")
    print("=" * 60)
    
    try:
        # Scrape active listings
        print("\n1. Scraping active listings...")
        active_properties = scraper.get_properties_api(status='active', max_homes=500)
        
        if active_properties:
            active_df = scraper.create_ml_dataset(active_properties, 'active')
            active_file = scraper.save_dataset(active_df, 'cincinnati_active_listings')
            
            print(f"   ✓ Collected {len(active_df)} active listings")
            print(f"   ✓ Saved to: {active_file}")
        
        # Scrape recently sold properties
        print("\n2. Scraping recently sold properties...")
        sold_properties = scraper.get_properties_api(status='sold', max_homes=1000)
        
        if sold_properties:
            sold_df = scraper.create_ml_dataset(sold_properties, 'sold')
            sold_file = scraper.save_dataset(sold_df, 'cincinnati_sold_properties')
            
            print(f"   ✓ Collected {len(sold_df)} sold properties")
            print(f"   ✓ Saved to: {sold_file}")
        
        # Create combined dataset
        if active_properties and sold_properties:
            print("\n3. Creating combined ML dataset...")
            
            # Add target variable (0 = active, 1 = sold)
            active_df['target_sold'] = 0
            sold_df['target_sold'] = 1
            
            # Combine datasets
            combined_df = pd.concat([active_df, sold_df], ignore_index=True)
            combined_file = scraper.save_dataset(combined_df, 'cincinnati_ml_dataset')
            
            print(f"   ✓ Combined dataset: {len(combined_df)} total properties")
            print(f"   ✓ Saved to: {combined_file}")
            
            # Generate summary statistics
            print("\n4. Generating summary statistics...")
            stats = scraper.generate_summary_stats(combined_df)
            
            # Save stats
            stats_file = scraper.save_dataset(pd.DataFrame([stats]), 'cincinnati_market_stats', 'json')
            
            print(f"\n" + "=" * 60)
            print("DATA EXTRACTION COMPLETE")
            print("=" * 60)
            
            print(f"\n📊 Dataset Summary:")
            print(f"   • Total properties: {stats['total_properties']}")
            print(f"   • Active listings: {len(active_df)}")
            print(f"   • Sold properties: {len(sold_df)}")
            
            if stats['price_stats']['median']:
                print(f"   • Median price: ${stats['price_stats']['median']:,.0f}")
                print(f"   • Price range: ${stats['price_stats']['min']:,.0f} - ${stats['price_stats']['max']:,.0f}")
            
            print(f"\n🏠 Top property types:")
            for prop_type, count in list(stats['property_types'].items())[:3]:
                print(f"   • {prop_type}: {count}")
            
            print(f"\n✅ Ready for ML model training!")
            print(f"✅ Dataset optimized with derived features")
            print(f"✅ Target variable (sold/active) included")
        
        else:
            print("\n❌ Failed to collect sufficient data")
    
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()