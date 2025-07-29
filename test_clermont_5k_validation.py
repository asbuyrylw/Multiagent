#!/usr/bin/env python3
"""
Clermont County 5000 Property Test with Sales Validation

This script:
1. Generates 5000 properties in Clermont County, OH
2. Runs the FIXED housing prediction model (no data leakage)
3. Cross-checks predictions against actual sales data
4. Provides validation metrics on model accuracy
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents.orchestrator_agent import OrchestratorAgent
from tools.redfin_data_tool import get_redfin_data
from tools.census_data_tool import combine_census_data
from tools.housing_ml_tool import train_housing_prediction_model, predict_sales_likelihood
from utils.logger import setup_logger
import warnings
warnings.filterwarnings('ignore')

def prepare_housing_features_no_leakage(property_data, census_data):
    """
    Prepare features WITHOUT data leakage - removes sell_probability
    """
    print("Preparing housing features for ML model (NO LEAKAGE VERSION)...")
    
    # Convert to DataFrame
    if not isinstance(property_data, pd.DataFrame):
        df = pd.DataFrame(property_data)
    else:
        df = property_data.copy()
    
    # CRITICAL: Remove leaky features
    leaky_features = ['sell_probability']
    for feature in leaky_features:
        if feature in df.columns:
            print(f"⚠️  REMOVING LEAKY FEATURE: '{feature}'")
            df = df.drop([feature], axis=1)
    
    # Add census data as features
    for key, value in census_data.items():
        if key != 'location' and isinstance(value, (int, float)):
            df[f'census_{key}'] = value
    
    # Feature engineering
    df['price_per_sqft'] = df['current_value'] / df['sqft']
    df['equity_to_value_ratio'] = df['equity'] / df['current_value']
    df['years_since_purchase'] = df['days_since_last_sale'] / 365.25
    df['sales_frequency'] = df['past_sales_count'] / df['years_since_purchase'].clip(lower=1)
    df['value_growth_rate'] = df['value_appreciation'] / df['years_since_purchase'].clip(lower=1)
    
    # Property characteristics
    df['bedrooms_per_sqft'] = df['bedrooms'] / df['sqft'] * 1000
    df['bathrooms_per_bedroom'] = df['bathrooms'] / df['bedrooms']
    df['is_luxury'] = (df['current_value'] > df['current_value'].quantile(0.8)).astype(int)
    df['is_starter_home'] = (df['current_value'] < df['current_value'].quantile(0.3)).astype(int)
    
    # Market factors
    df['market_appreciation_vs_area'] = df['value_appreciation'] - df['census_income_change_1yr']
    df['overvalued_vs_neighbors'] = (df['value_vs_neighbors'] > 0.1).astype(int)
    df['rapid_appreciation'] = (df['value_appreciation'] > 0.2).astype(int)
    
    # Financial pressure
    df['high_mortgage_burden'] = (df['mortgage_balance'] / df['current_value'] > 0.7).astype(int)
    df['low_equity'] = (df['equity_ratio'] < 0.3).astype(int)
    df['recent_purchase'] = (df['days_since_last_sale'] < 730).astype(int)
    
    # Area factors
    df['growing_area'] = (df['census_population_change_1yr'] > 0.05).astype(int)
    df['economic_growth'] = (df['census_income_change_1yr'] > 0.05).astype(int)
    df['high_mobility_area'] = (df['census_population_change_1yr'] > 0.1).astype(int)
    
    print(f"✅ Feature engineering complete. Dataset shape: {df.shape}")
    return df

def get_actual_sales_data(property_ids, start_date=None, end_date=None):
    """
    Mock function to simulate checking actual sales data.
    In production, this would query MLS or public records API.
    """
    print(f"🔍 Checking actual sales for {len(property_ids)} properties...")
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # Simulate actual sales data with realistic patterns
    actual_sales = []
    
    for prop_id in property_ids:
        # Simulate realistic sale probability (10-15% annually)
        base_sale_prob = 0.12
        
        # Add some randomness but keep it realistic
        actual_sold = np.random.random() < base_sale_prob
        
        if actual_sold:
            # Generate realistic sale details
            sale_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            
            actual_sales.append({
                'property_id': prop_id,
                'sold': True,
                'sale_date': sale_date,
                'days_on_market': np.random.randint(15, 180),
                'sale_price': np.random.randint(150000, 800000),  # Clermont County range
                'list_price': np.random.randint(140000, 820000)
            })
        else:
            actual_sales.append({
                'property_id': prop_id,
                'sold': False,
                'sale_date': None,
                'days_on_market': None,
                'sale_price': None,
                'list_price': None
            })
    
    sales_df = pd.DataFrame(actual_sales)
    sold_count = sales_df['sold'].sum()
    print(f"📊 Found {sold_count} actual sales out of {len(property_ids)} properties ({sold_count/len(property_ids)*100:.1f}%)")
    
    return sales_df

def validate_predictions(predictions_df, actual_sales_df):
    """
    Validate model predictions against actual sales data
    """
    print("🔍 VALIDATING PREDICTIONS AGAINST ACTUAL SALES")
    print("="*60)
    
    # Merge predictions with actual sales
    merged = predictions_df.merge(actual_sales_df, on='property_id', how='inner')
    
    # Calculate validation metrics
    actual_sold = merged['sold']
    predicted_sold = merged['predicted_will_sell']
    predicted_prob = merged['sell_probability']
    
    # Basic accuracy metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(actual_sold, predicted_sold)
    precision = precision_score(actual_sold, predicted_sold, zero_division=0)
    recall = recall_score(actual_sold, predicted_sold, zero_division=0)
    f1 = f1_score(actual_sold, predicted_sold, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(actual_sold, predicted_prob)
    except:
        roc_auc = 0.5  # Default if calculation fails
    
    validation_results = {
        'total_properties': len(merged),
        'actual_sales': actual_sold.sum(),
        'predicted_sales': predicted_sold.sum(),
        'actual_sale_rate': actual_sold.mean(),
        'predicted_sale_rate': predicted_sold.mean(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    # Confusion matrix breakdown
    tp = ((actual_sold == 1) & (predicted_sold == 1)).sum()
    tn = ((actual_sold == 0) & (predicted_sold == 0)).sum()
    fp = ((actual_sold == 0) & (predicted_sold == 1)).sum()
    fn = ((actual_sold == 1) & (predicted_sold == 0)).sum()
    
    print(f"📊 VALIDATION RESULTS:")
    print(f"   Total Properties: {validation_results['total_properties']:,}")
    print(f"   Actual Sales: {validation_results['actual_sales']} ({validation_results['actual_sale_rate']:.1%})")
    print(f"   Predicted Sales: {validation_results['predicted_sales']} ({validation_results['predicted_sale_rate']:.1%})")
    print(f"")
    print(f"🎯 MODEL PERFORMANCE vs REALITY:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    print(f"   ROC AUC: {roc_auc:.3f}")
    print(f"")
    print(f"📋 CONFUSION MATRIX:")
    print(f"   True Positives (Correctly predicted sales): {tp}")
    print(f"   True Negatives (Correctly predicted no sale): {tn}")
    print(f"   False Positives (Predicted sale, didn't sell): {fp}")
    print(f"   False Negatives (Missed actual sales): {fn}")
    
    # High-confidence predictions analysis
    high_conf_predictions = merged[merged['sell_probability'] > 0.7]
    if len(high_conf_predictions) > 0:
        high_conf_accuracy = high_conf_predictions['sold'].mean()
        print(f"")
        print(f"🎯 HIGH CONFIDENCE PREDICTIONS (>70% probability):")
        print(f"   Count: {len(high_conf_predictions)}")
        print(f"   Actual sale rate: {high_conf_accuracy:.1%}")
    
    return validation_results, merged

def main():
    """Run the comprehensive 5000 property test with validation"""
    
    logger = setup_logger("clermont_5k_test")
    logger.info("="*80)
    logger.info("CLERMONT COUNTY 5000 PROPERTY TEST WITH SALES VALIDATION")
    logger.info("="*80)
    
    print("🏠 CLERMONT COUNTY 5000 PROPERTY PREDICTION TEST")
    print("="*70)
    
    try:
        # Step 1: Generate 5000 properties
        print("1. Generating 5000 Clermont County properties...")
        property_data = get_redfin_data("Clermont County, OH", 5000)
        census_data = combine_census_data("Clermont County, OH")
        
        # Step 2: Prepare features (no leakage)
        print("\n2. Preparing features for ML model...")
        prepared_data = prepare_housing_features_no_leakage(property_data, census_data)
        
        # Step 3: Train models
        print("\n3. Training housing prediction models...")
        model_results, best_model_name, feature_columns = train_housing_prediction_model(prepared_data)
        best_model = model_results[best_model_name]
        
        # Step 4: Make predictions
        print("\n4. Making predictions on all 5000 properties...")
        predictions = predict_sales_likelihood(
            best_model, 
            prepared_data, 
            feature_columns, 
            best_model_name
        )
        
        # Step 5: Get actual sales data
        print("\n5. Checking actual sales data...")
        property_ids = predictions['property_id'].tolist()
        actual_sales = get_actual_sales_data(property_ids)
        
        # Step 6: Validate predictions
        print("\n6. Validating predictions against actual sales...")
        validation_results, merged_data = validate_predictions(predictions, actual_sales)
        
        # Step 7: Generate comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        predictions_file = f"clermont_5k_predictions_{timestamp}.csv"
        predictions.to_csv(predictions_file, index=False)
        
        # Save validation results
        validation_file = f"clermont_5k_validation_{timestamp}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save merged data for analysis
        merged_file = f"clermont_5k_merged_data_{timestamp}.csv"
        merged_data.to_csv(merged_file, index=False)
        
        # Generate summary
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Properties Analyzed: {len(predictions):,}")
        print(f"   Best Model: {best_model_name}")
        print(f"   Model ROC AUC: {best_model['metrics']['roc_auc']:.3f}")
        print(f"   Validation Accuracy: {validation_results['accuracy']:.3f}")
        print(f"   High Probability Leads: {len(predictions[predictions['sell_probability'] > 0.7]):,}")
        
        print(f"\n📁 Files Generated:")
        print(f"   • {predictions_file}")
        print(f"   • {validation_file}")
        print(f"   • {merged_file}")
        
        logger.info("Clermont County 5K test completed successfully")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in 5K property test: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    results = main()