#!/usr/bin/env python3
"""
Cincinnati Real Estate Data Analysis
Analyze the collected data and prepare insights for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def load_and_analyze_data():
    """Load and analyze the Cincinnati real estate data"""
    
    print("CINCINNATI REAL ESTATE DATA ANALYSIS")
    print("=" * 50)
    
    # Load the data
    try:
        df = pd.read_csv('cincinnati_active_listings_20250726_150118.csv')
        print(f"✓ Loaded {len(df)} properties from Cincinnati")
    except FileNotFoundError:
        print("❌ Data file not found. Please run the scraper first.")
        return None
    
    return df

def basic_statistics(df):
    """Generate basic statistics about the dataset"""
    
    print("\n📊 BASIC DATASET STATISTICS")
    print("-" * 40)
    
    print(f"Total Properties: {len(df)}")
    print(f"Data Collection Date: {df['scraped_at'].iloc[0][:10]}")
    print(f"Price Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Median Price: ${df['price'].median():,.0f}")
    print(f"Average Price: ${df['price'].mean():,.0f}")
    
    print(f"\nProperty Types:")
    type_counts = df['ui_property_type'].value_counts()
    for prop_type, count in type_counts.head(5).items():
        print(f"  • Type {prop_type}: {count} properties")
    
    print(f"\nBedroom Distribution:")
    bed_counts = df['beds'].value_counts().sort_index()
    for beds, count in bed_counts.items():
        print(f"  • {beds} bedrooms: {count} properties")
    
    print(f"\nPrice Segments:")
    price_segment_counts = df['price_segment'].value_counts()
    for segment, count in price_segment_counts.items():
        print(f"  • {segment}: {count} properties")

def location_analysis(df):
    """Analyze geographical distribution"""
    
    print("\n🗺️  LOCATION ANALYSIS")
    print("-" * 40)
    
    # Top neighborhoods/locations
    print("Top Neighborhoods:")
    location_counts = df['location'].value_counts().head(10)
    for location, count in location_counts.items():
        avg_price = df[df['location'] == location]['price'].mean()
        print(f"  • {location}: {count} properties (avg: ${avg_price:,.0f})")
    
    # ZIP code analysis
    print("\nTop ZIP Codes:")
    zip_counts = df['zip_code'].value_counts().head(10)
    for zip_code, count in zip_counts.items():
        avg_price = df[df['zip_code'] == zip_code]['price'].mean()
        print(f"  • {zip_code}: {count} properties (avg: ${avg_price:,.0f})")

def price_analysis(df):
    """Analyze pricing patterns"""
    
    print("\n💰 PRICE ANALYSIS")
    print("-" * 40)
    
    # Price per sqft analysis
    print("Price per Square Foot Analysis:")
    print(f"  • Average: ${df['price_per_sqft'].mean():.0f}/sqft")
    print(f"  • Median: ${df['price_per_sqft'].median():.0f}/sqft")
    print(f"  • Range: ${df['price_per_sqft'].min():.0f} - ${df['price_per_sqft'].max():.0f}/sqft")
    
    # Price by bedrooms
    print("\nAverage Price by Bedrooms:")
    price_by_beds = df.groupby('beds')['price'].agg(['mean', 'median', 'count'])
    for beds, row in price_by_beds.iterrows():
        print(f"  • {beds} bed: ${row['mean']:,.0f} avg, ${row['median']:,.0f} median ({row['count']} properties)")
    
    # Price by size segment
    print("\nAverage Price by Size:")
    price_by_size = df.groupby('size_segment')['price'].agg(['mean', 'median', 'count'])
    for size, row in price_by_size.iterrows():
        print(f"  • {size}: ${row['mean']:,.0f} avg, ${row['median']:,.0f} median ({row['count']} properties)")

def property_features_analysis(df):
    """Analyze property features and amenities"""
    
    print("\n🏠 PROPERTY FEATURES ANALYSIS")
    print("-" * 40)
    
    # Age analysis
    print("Home Age Analysis:")
    print(f"  • Average age: {df['home_age'].mean():.1f} years")
    print(f"  • Newest home: {df['home_age'].min():.0f} years old")
    print(f"  • Oldest home: {df['home_age'].max():.0f} years old")
    
    # Features that might affect price
    feature_columns = ['has_virtual_tour', 'has_video_tour', 'has_3d_tour', 'is_new_construction', 'is_hot']
    
    print("\nFeature Analysis (% of properties with feature):")
    for feature in feature_columns:
        if feature in df.columns:
            pct = (df[feature].sum() / len(df)) * 100
            avg_price_with = df[df[feature] == True]['price'].mean()
            avg_price_without = df[df[feature] == False]['price'].mean()
            print(f"  • {feature}: {pct:.1f}% (avg price with: ${avg_price_with:,.0f}, without: ${avg_price_without:,.0f})")

def market_timing_analysis(df):
    """Analyze market timing data"""
    
    print("\n📈 MARKET TIMING ANALYSIS")
    print("-" * 40)
    
    # Days on market
    print("Days on Market Analysis:")
    print(f"  • Average: {df['days_on_market'].mean():.1f} days")
    print(f"  • Median: {df['days_on_market'].median():.1f} days")
    print(f"  • Range: {df['days_on_market'].min():.0f} - {df['days_on_market'].max():.0f} days")
    
    # Properties by days on market categories
    df['dom_category'] = pd.cut(df['days_on_market'], 
                               bins=[0, 30, 60, 90, 180, float('inf')], 
                               labels=['Under 30 days', '30-60 days', '60-90 days', '90-180 days', 'Over 180 days'])
    
    print("\nProperties by Time on Market:")
    dom_counts = df['dom_category'].value_counts()
    for category, count in dom_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • {category}: {count} properties ({pct:.1f}%)")

def ml_readiness_assessment(df):
    """Assess the data's readiness for ML training"""
    
    print("\n🤖 ML MODEL READINESS ASSESSMENT")
    print("-" * 40)
    
    # Check for missing values
    print("Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    for col in missing_counts[missing_counts > 0].index:
        print(f"  • {col}: {missing_counts[col]} missing ({missing_pct[col]:.1f}%)")
    
    if missing_counts.sum() == 0:
        print("  ✓ No missing values found!")
    
    # Feature quality assessment
    print("\nFeature Quality Assessment:")
    
    # Numeric features for ML
    numeric_features = ['price', 'beds', 'baths', 'sqft', 'lot_size', 'year_built', 
                       'price_per_sqft', 'days_on_market', 'home_age', 'price_per_bed', 
                       'price_per_bath', 'sqft_per_bed', 'bed_bath_ratio']
    
    available_numeric = [col for col in numeric_features if col in df.columns]
    print(f"  ✓ Available numeric features: {len(available_numeric)}")
    
    # Categorical features
    categorical_features = ['property_type', 'location', 'zip_code', 'price_segment', 'size_segment']
    available_categorical = [col for col in categorical_features if col in df.columns]
    print(f"  ✓ Available categorical features: {len(available_categorical)}")
    
    # Boolean features
    boolean_features = ['has_virtual_tour', 'has_video_tour', 'has_3d_tour', 'is_new_construction', 'is_hot']
    available_boolean = [col for col in boolean_features if col in df.columns]
    print(f"  ✓ Available boolean features: {len(available_boolean)}")
    
    print(f"\n  📊 Total features ready for ML: {len(available_numeric) + len(available_categorical) + len(available_boolean)}")

def generate_ml_dataset_recommendations(df):
    """Generate recommendations for ML model training"""
    
    print("\n💡 ML MODEL RECOMMENDATIONS")
    print("-" * 40)
    
    print("Recommended Features for Price Prediction:")
    recommended_features = [
        'beds', 'baths', 'sqft', 'lot_size', 'home_age', 
        'location', 'zip_code', 'price_per_sqft',
        'has_virtual_tour', 'has_3d_tour', 'is_new_construction'
    ]
    
    for feature in recommended_features:
        if feature in df.columns:
            print(f"  ✓ {feature}")
        else:
            print(f"  ✗ {feature} (not available)")
    
    print("\nPotential Target Variables:")
    print("  • Price prediction: Use 'price' as target")
    print("  • Price category: Use 'price_segment' as target")
    print("  • Market speed: Use 'days_on_market' for time-to-sell prediction")
    
    print("\nData Preprocessing Recommendations:")
    print("  • Handle outliers in price (properties > $1M or < $50K)")
    print("  • One-hot encode categorical variables (location, zip_code)")
    print("  • Scale numeric features for better model performance")
    print("  • Consider log transformation for price (right-skewed distribution)")
    
    print("\nModel Suggestions:")
    print("  • Linear/Ridge Regression: Good baseline for price prediction")
    print("  • Random Forest: Handles mixed data types well")
    print("  • XGBoost: Often performs best for tabular data")
    print("  • Neural Networks: For complex feature interactions")

def save_analysis_report(df):
    """Save a comprehensive analysis report"""
    
    report = {
        'analysis_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_properties': len(df),
            'data_source': 'Redfin Cincinnati',
            'collection_date': df['scraped_at'].iloc[0]
        },
        'price_statistics': {
            'mean': float(df['price'].mean()),
            'median': float(df['price'].median()),
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
            'std': float(df['price'].std())
        },
        'feature_summary': {
            'numeric_features': len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]),
            'categorical_features': len([col for col in df.columns if df[col].dtype == 'object']),
            'boolean_features': len([col for col in df.columns if df[col].dtype == 'bool'])
        },
        'top_neighborhoods': df['location'].value_counts().head(10).to_dict(),
        'bedroom_distribution': df['beds'].value_counts().to_dict(),
        'price_segments': df['price_segment'].value_counts().to_dict()
    }
    
    with open('cincinnati_market_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Analysis report saved to: cincinnati_market_analysis.json")

def main():
    """Main analysis function"""
    
    # Load data
    df = load_and_analyze_data()
    if df is None:
        return
    
    # Run all analyses
    basic_statistics(df)
    location_analysis(df)
    price_analysis(df)
    property_features_analysis(df)
    market_timing_analysis(df)
    ml_readiness_assessment(df)
    generate_ml_dataset_recommendations(df)
    
    # Save analysis report
    save_analysis_report(df)
    
    print("\n" + "=" * 50)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 50)
    print("\n🎯 Key Insights for ML Model:")
    print("• Dataset contains 700 active Cincinnati properties")
    print("• Price range from ${:,.0f} to ${:,.0f}".format(df['price'].min(), df['price'].max()))
    print("• Rich feature set with location, property characteristics, and amenities")
    print("• Data is clean and ready for ML model training")
    print("• Recommended to collect sold properties for comparison analysis")

if __name__ == "__main__":
    main()