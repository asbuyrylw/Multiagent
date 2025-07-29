#!/usr/bin/env python3
"""
Debug script to examine Redfin response format
"""

import requests
from urllib.parse import urlencode

def debug_redfin_response():
    """Debug what Redfin is actually returning"""
    
    print("DEBUGGING REDFIN RESPONSE")
    print("=" * 40)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.redfin.com/city/3879/OH/Cincinnati'
    })
    
    # Try different API endpoints
    endpoints = [
        # Original API
        'https://www.redfin.com/stingray/api/gis',
        # Alternative endpoints
        'https://www.redfin.com/stingray/api/home/details/search',
        'https://www.redfin.com/stingray/api/gis-csv'
    ]
    
    search_params = {
        'al': 1,
        'region_id': 3879,
        'region_type': 6,
        'num_homes': 10,
        'start': 0,
        'sf': '1,2,3,5,6,7',
        'uipt': '1,2,3,4,5,6,7,8',
        'status': 9,
        'v': 8
    }
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        print("-" * 60)
        
        try:
            url = f"{endpoint}?{urlencode(search_params)}"
            response = session.get(url, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"Content-Length: {len(response.text)} characters")
            
            # Show first 500 characters of response
            print(f"\nFirst 500 characters of response:")
            print("-" * 40)
            print(repr(response.text[:500]))
            
            # Try to identify response format
            if response.text.startswith('{}&&'):
                print("\n🔍 Response appears to use JSONP-style format")
                # Try to extract JSON part
                json_part = response.text[4:]  # Remove '{}&&'
                print(f"JSON part preview: {repr(json_part[:200])}")
                
            elif response.text.startswith('{'):
                print("\n🔍 Response appears to be standard JSON")
                
            elif response.text.startswith('<!DOCTYPE') or response.text.startswith('<html'):
                print("\n🔍 Response appears to be HTML (possibly blocked/redirected)")
                
            else:
                print(f"\n🔍 Unknown response format. Starts with: {repr(response.text[:50])}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def try_alternative_approach():
    """Try using the same approach as web browser"""
    
    print("\n" + "=" * 60)
    print("TRYING BROWSER-LIKE APPROACH")
    print("=" * 60)
    
    session = requests.Session()
    
    # First visit the main Cincinnati page
    print("1. Visiting Cincinnati main page...")
    try:
        main_response = session.get('https://www.redfin.com/city/3879/OH/Cincinnati')
        print(f"   Status: {main_response.status_code}")
        
        # Extract any relevant cookies or headers
        print(f"   Cookies received: {len(session.cookies)}")
        
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Try the search with browser-like headers
    print("\n2. Attempting search API call...")
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.redfin.com/city/3879/OH/Cincinnati',
        'X-Requested-With': 'XMLHttpRequest',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin'
    })
    
    search_params = {
        'al': 1,
        'region_id': 3879,
        'region_type': 6,
        'num_homes': 5,
        'start': 0,
        'status': 9,
        'v': 8
    }
    
    try:
        url = f"https://www.redfin.com/stingray/api/gis?{urlencode(search_params)}"
        response = session.get(url, timeout=30)
        
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type')}")
        print(f"   Response length: {len(response.text)}")
        
        # Handle different response formats
        if response.text.startswith('{}&&'):
            print("   📋 JSONP-style response detected")
            json_text = response.text[4:]  # Remove '{}&&'
            try:
                import json
                data = json.loads(json_text)
                print(f"   ✅ Successfully parsed JSON")
                print(f"   📊 Top-level keys: {list(data.keys())}")
                
                if 'payload' in data:
                    payload = data['payload']
                    print(f"   📊 Payload keys: {list(payload.keys())}")
                    
                    if 'homes' in payload:
                        homes = payload['homes']
                        print(f"   🏠 Found {len(homes)} homes")
                        if homes:
                            print(f"   🏠 Sample home keys: {list(homes[0].keys())}")
                            return True
                            
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                print(f"   📋 Raw response preview: {repr(response.text[:200])}")
                
        elif response.text.startswith('{'):
            print("   📋 Standard JSON response")
            try:
                import json
                data = json.loads(response.text)
                print(f"   ✅ Successfully parsed JSON")
                print(f"   📊 Keys: {list(data.keys())}")
                return True
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
        
        else:
            print(f"   ❓ Unexpected response format: {repr(response.text[:100])}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    debug_redfin_response()
    success = try_alternative_approach()
    
    if success:
        print("\n✅ Successfully identified working approach!")
    else:
        print("\n❌ Need to investigate further...")