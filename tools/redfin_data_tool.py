import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import random

def get_redfin_data(location, property_count=100):
    """
    Simulate fetching Redfin housing data for a given location.
    In production, this would integrate with Redfin's API or web scraping.
    """
    print(f"Fetching Redfin data for {location}...")
    
    # Simulate API delay
    time.sleep(2)
    
    # Generate mock housing data
    properties = []
    
    for i in range(property_count):
        # Property characteristics
        bedrooms = random.choice([2, 3, 4, 5, 6])
        bathrooms = random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4])
        sqft = random.randint(800, 4500)
        
        # Set location-specific base values
        if "cincinnati" in location.lower() or "ohio" in location.lower():
            base_lat, base_lon = 39.1031, -84.5120  # Cincinnati coordinates
            base_value_range = (150000, 800000)
        else:
            base_lat, base_lon = 47.6062, -122.3321  # Default to Seattle
            base_value_range = (400000, 1500000)
            
        # Location factors
        lat = base_lat + random.uniform(-0.3, 0.3)
        lon = base_lon + random.uniform(-0.3, 0.3)
        
        # Time-based factors
        days_since_last_sale = random.randint(30, 3650)  # 1 month to 10 years
        past_sales_count = random.randint(1, 8)
        
        # Property values with more realistic variation
        base_value = random.randint(base_value_range[0], base_value_range[1])
        # Value based on size and features (but not perfectly correlated)
        size_multiplier = 0.7 + (sqft / 4500) * 0.6  # 0.7 to 1.3 range
        feature_multiplier = 1 + (bedrooms - 2) * 0.05 + (bathrooms - 1) * 0.03
        
        current_value = int(base_value * size_multiplier * feature_multiplier * random.uniform(0.85, 1.15))
        purchase_price = int(current_value * random.uniform(0.6, 1.2))  # More realistic price variation
        
        if purchase_price > 0:
            value_appreciation = (current_value - purchase_price) / purchase_price
        else:
            value_appreciation = 0
        
        # Neighborhood factors (mock)
        avg_neighbor_appreciation = random.uniform(-0.1, 0.4)
        value_vs_neighbors = value_appreciation - avg_neighbor_appreciation
        
        # Financial factors
        mortgage_balance = random.randint(0, int(current_value * 0.8))
        equity = current_value - mortgage_balance
        equity_ratio = equity / current_value
        
        # Demographic movement (mock census-like data)
        area_population_change = random.uniform(-0.05, 0.15)
        area_income_change = random.uniform(-0.03, 0.12)
        
        # FIXED: Create more realistic likelihood of selling with proper noise and independence
        # Base probability influenced by market conditions and random factors
        base_sell_prob = 0.15  # Base 15% chance per year
        
        # Market timing factors (seasonal, economic cycles)
        market_timing_factor = random.uniform(0.8, 1.2)
        
        # Life events (job changes, family size, age-related moves) - random
        life_event_factor = random.uniform(0.5, 2.0)
        
        # Economic pressure (independent of our features to avoid leakage)
        economic_pressure = random.uniform(0.7, 1.5)
        
        # Personal factors (moving for lifestyle, etc.)
        personal_factor = random.uniform(0.6, 1.8)
        
        # Calculate final probability with substantial randomness
        sell_probability = base_sell_prob * market_timing_factor * life_event_factor * economic_pressure * personal_factor
        
        # Add significant noise to break any remaining correlations
        noise = random.normalvariate(0, 0.15)  # Normal distribution noise
        sell_probability += noise
        
        # Clamp to realistic range
        sell_probability = max(0.01, min(0.95, sell_probability))
        
        # Binary target with variable threshold for more realism
        will_sell = sell_probability > random.uniform(0.4, 0.7)
        
        property_data = {
            'property_id': f'RF_{location}_{i+1:04d}',
            'address': f'{random.randint(100, 9999)} {random.choice(["Main", "Oak", "Pine", "Cedar", "Maple"])} {random.choice(["St", "Ave", "Rd", "Blvd"])}',
            'latitude': lat,
            'longitude': lon,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft': sqft,
            'current_value': current_value,
            'purchase_price': purchase_price,
            'value_appreciation': value_appreciation,
            'value_vs_neighbors': value_vs_neighbors,
            'days_since_last_sale': days_since_last_sale,
            'past_sales_count': past_sales_count,
            'mortgage_balance': mortgage_balance,
            'equity': equity,
            'equity_ratio': equity_ratio,
            'area_population_change': area_population_change,
            'area_income_change': area_income_change,
            'sell_probability': sell_probability,
            'will_sell_within_year': will_sell
        }
        
        properties.append(property_data)
    
    df = pd.DataFrame(properties)
    
    print(f"Successfully fetched {len(df)} properties from Redfin for {location}")
    print(f"Properties likely to sell: {df['will_sell_within_year'].sum()}")
    
    return df

def get_property_details(property_id):
    """
    Get detailed information for a specific property.
    Mock implementation for testing.
    """
    print(f"Fetching detailed data for property {property_id}...")
    
    # In production, this would make specific API calls for property history
    return {
        'property_id': property_id,
        'listing_history': random.randint(1, 10),
        'price_changes': random.randint(0, 5),
        'days_on_market_last_sale': random.randint(10, 200),
        'neighborhood_sales_last_6_months': random.randint(5, 50),
        'school_rating': random.randint(1, 10),
        'walkability_score': random.randint(10, 100)
    }
