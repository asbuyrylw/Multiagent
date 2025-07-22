import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

def get_census_data(location, radius_miles=5):
    """
    Simulate fetching US Census demographic and economic data for an area.
    In production, this would use the US Census API.
    """
    print(f"Fetching Census data for {location} within {radius_miles} miles...")
    
    # Simulate API delay
    time.sleep(1.5)
    
    # Generate mock census data
    census_data = {
        'location': location,
        'total_population': random.randint(10000, 500000),
        'population_change_1yr': random.uniform(-0.05, 0.15),
        'population_change_5yr': random.uniform(-0.1, 0.25),
        'median_age': random.uniform(25, 55),
        'median_household_income': random.randint(35000, 150000),
        'income_change_1yr': random.uniform(-0.05, 0.12),
        'income_change_5yr': random.uniform(-0.1, 0.3),
        'unemployment_rate': random.uniform(2, 12),
        'education_bachelor_plus': random.uniform(15, 75),  # Percentage
        'homeownership_rate': random.uniform(40, 85),  # Percentage
        'rental_rate': random.uniform(15, 60),  # Percentage
        'housing_units_total': random.randint(4000, 200000),
        'housing_units_change_1yr': random.uniform(-0.03, 0.08),
        'housing_units_change_5yr': random.uniform(-0.05, 0.2),
        'new_construction_permits': random.randint(0, 1000),
        'business_establishments': random.randint(500, 15000),
        'business_change_1yr': random.uniform(-0.1, 0.15),
        'commute_time_avg': random.uniform(15, 45),  # Minutes
        'public_transport_usage': random.uniform(2, 35),  # Percentage
        'crime_rate_per_1000': random.uniform(5, 50),
        'school_district_rating': random.randint(1, 10)
    }
    
    print(f"Successfully fetched Census data for {location}")
    return census_data

def get_migration_patterns(location):
    """
    Simulate fetching migration and mobility data from Census.
    """
    print(f"Fetching migration patterns for {location}...")
    
    time.sleep(1)
    
    # Generate mock migration data
    migration_data = {
        'location': location,
        'in_migration_rate': random.uniform(5, 25),  # Per 1000 residents
        'out_migration_rate': random.uniform(5, 25),
        'net_migration_rate': random.uniform(-10, 15),
        'interstate_moves_in': random.randint(100, 5000),
        'interstate_moves_out': random.randint(100, 5000),
        'intrastate_moves_in': random.randint(200, 8000),
        'intrastate_moves_out': random.randint(200, 8000),
        'international_moves_in': random.randint(50, 2000),
        'avg_resident_tenure': random.uniform(3, 15),  # Years
        'mobility_index': random.uniform(0.1, 0.4),  # Higher = more mobile
        'job_growth_rate': random.uniform(-0.05, 0.15),
        'major_employers_change': random.randint(-3, 8)  # Net change in major employers
    }
    
    print(f"Successfully fetched migration data for {location}")
    return migration_data

def get_economic_indicators(location):
    """
    Simulate fetching economic indicators from various Census surveys.
    """
    print(f"Fetching economic indicators for {location}...")
    
    time.sleep(1)
    
    economic_data = {
        'location': location,
        'gdp_per_capita': random.randint(30000, 120000),
        'gdp_growth_rate': random.uniform(-0.05, 0.08),
        'cost_of_living_index': random.uniform(85, 150),  # 100 = national average
        'housing_cost_burden': random.uniform(20, 50),  # % of income on housing
        'property_tax_rate': random.uniform(0.5, 2.5),  # Percentage
        'local_tax_burden': random.uniform(8, 15),  # % of income
        'retail_sales_growth': random.uniform(-0.1, 0.2),
        'commercial_development': random.randint(0, 20),  # New projects
        'infrastructure_investment': random.randint(0, 100),  # Million $
        'zoning_changes_residential': random.randint(0, 10),
        'market_volatility_index': random.uniform(0.1, 0.8)  # 0=stable, 1=volatile
    }
    
    print(f"Successfully fetched economic data for {location}")
    return economic_data

def combine_census_data(location):
    """
    Combine all census data sources into a comprehensive dataset.
    """
    print(f"Combining all Census data for {location}...")
    
    basic_data = get_census_data(location)
    migration_data = get_migration_patterns(location)
    economic_data = get_economic_indicators(location)
    
    # Combine all data
    combined_data = {**basic_data, **migration_data, **economic_data}
    
    # Calculate derived metrics
    combined_data['population_stability'] = 1 - combined_data['mobility_index']
    combined_data['economic_health'] = (
        (combined_data['income_change_1yr'] + 0.05) * 0.3 +
        (combined_data['job_growth_rate'] + 0.05) * 0.3 +
        (combined_data['gdp_growth_rate'] + 0.05) * 0.2 +
        (1 - combined_data['unemployment_rate'] / 15) * 0.2
    )
    combined_data['area_attractiveness'] = (
        combined_data['population_change_1yr'] * 0.2 +
        combined_data['income_change_1yr'] * 0.2 +
        combined_data['job_growth_rate'] * 0.2 +
        (combined_data['school_district_rating'] / 10) * 0.2 +
        (1 - combined_data['crime_rate_per_1000'] / 50) * 0.2
    )
    
    print(f"Successfully combined all Census data for {location}")
    return combined_data
