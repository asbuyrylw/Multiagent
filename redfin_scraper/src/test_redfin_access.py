#!/usr/bin/env python3
"""
Test script to verify Redfin data access for Cincinnati
"""

import requests
import json
from urllib.parse import urlencode

def test_redfin_access():
    """Test basic access to Redfin Cincinnati data"""
    
    print("Testing Redfin Cincinnati Data Access...")
    print("=" * 50)
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.redfin.com/city/3879/OH/Cincinnati'
    })
    
    # Test 1: Access Cincinnati city page
    print("1. Testing Cincinnati city page access...")
    try:
        response = session.get('https://www.redfin.com/city/3879/OH/Cincinnati', timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ Successfully accessed Cincinnati page")
        else:
            print("   ✗ Failed to access Cincinnati page")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Try to access search API
    print("\n2. Testing search API access...")
    
    # Cincinnati search parameters
    search_params = {
        'al': 1,
        'region_id': 3879,
        'region_type': 6,
        'num_homes': 50,
        'start': 0,
        'sf': '1,2,3,5,6,7',
        'uipt': '1,2,3,4,5,6,7,8',
        'status': 9,  # Active listings
        'v': 8
    }
    
    search_url = f"https://www.redfin.com/stingray/api/gis?{urlencode(search_params)}"
    
    try:
        response = session.get(search_url, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Successfully accessed search API")
            
            # Check if we have homes data
            if 'payload' in data and 'homes' in data['payload']:
                homes = data['payload']['homes']
                print(f"   ✓ Found {len(homes)} properties")
                
                # Show sample property data
                if homes:
                    sample_home = homes[0]
                    print(f"\n   Sample property data:")
                    print(f"   - Address: {sample_home.get('streetLine', {}).get('value', 'N/A')}")
                    print(f"   - Price: ${sample_home.get('price', {}).get('value', 'N/A'):,}")
                    print(f"   - Beds: {sample_home.get('beds', 'N/A')}")
                    print(f"   - Baths: {sample_home.get('baths', 'N/A')}")
                    print(f"   - Sqft: {sample_home.get('sqFt', {}).get('value', 'N/A')}")
                    print(f"   - Property ID: {sample_home.get('propertyId', 'N/A')}")
                    
                    return True
            else:
                print("   ✗ No homes data in response")
                print(f"   Response keys: {list(data.keys())}")
                return False
        else:
            print(f"   ✗ API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_sold_properties():
    """Test access to sold properties data"""
    
    print("\n3. Testing sold properties access...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': 'https://www.redfin.com/city/3879/OH/Cincinnati'
    })
    
    # Parameters for sold properties
    sold_params = {
        'al': 1,
        'region_id': 3879,
        'region_type': 6,
        'num_homes': 50,
        'start': 0,
        'sf': '1,2,3,5,6,7',
        'uipt': '1,2,3,4,5,6,7,8',
        'status': 2,  # Sold
        'sold_within_days': 90,  # Last 90 days
        'v': 8
    }
    
    search_url = f"https://www.redfin.com/stingray/api/gis?{urlencode(sold_params)}"
    
    try:
        response = session.get(search_url, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'payload' in data and 'homes' in data['payload']:
                homes = data['payload']['homes']
                print(f"   ✓ Found {len(homes)} sold properties")
                
                if homes:
                    sample_home = homes[0]
                    print(f"\n   Sample sold property:")
                    print(f"   - Address: {sample_home.get('streetLine', {}).get('value', 'N/A')}")
                    print(f"   - Sold Price: ${sample_home.get('price', {}).get('value', 'N/A'):,}")
                    print(f"   - Sold Date: {sample_home.get('soldDate', 'N/A')}")
                    print(f"   - Days on Market: {sample_home.get('dom', 'N/A')}")
                    
                    return True
            else:
                print("   ✗ No sold homes data found")
                return False
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    """Main test function"""
    
    print("REDFIN CINCINNATI ACCESS TEST")
    print("=" * 40)
    
    # Test basic access
    if not test_redfin_access():
        print("\n❌ Basic access test failed")
        return
    
    # Test sold properties
    if not test_sold_properties():
        print("\n❌ Sold properties test failed")
        return
    
    print("\n" + "=" * 40)
    print("✅ ALL TESTS PASSED!")
    print("✅ Redfin data is accessible")
    print("✅ Ready to run full scraper")
    print("=" * 40)

if __name__ == "__main__":
    main()