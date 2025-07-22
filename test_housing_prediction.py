#!/usr/bin/env python3
"""
Test script for the Housing Sales Prediction Multi-Agent System

This script demonstrates the complete workflow:
1. Collecting Redfin housing data
2. Gathering US Census demographic data  
3. Training ML models to predict sales likelihood
4. Generating actionable insights and recommendations
"""

import sys
import json
import pandas as pd
from datetime import datetime
from agents.orchestrator_agent import OrchestratorAgent
from utils.logger import setup_logger

def main():
    """Run the housing prediction test."""
    logger = setup_logger("housing_test")
    logger.info("="*60)
    logger.info("HOUSING SALES PREDICTION MULTI-AGENT SYSTEM TEST")
    logger.info("="*60)
    
    try:
        # Initialize the orchestrator with housing workflow
        workflow_file = "workflows/housing_prediction_workflow.yaml"
        logger.info(f"Loading workflow: {workflow_file}")
        
        orchestrator = OrchestratorAgent(workflow_file)
        
        # Run the workflow
        logger.info("Starting housing prediction workflow...")
        start_time = datetime.now()
        
        result = orchestrator.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Workflow completed in {duration:.2f} seconds")
        
        # Process and display results
        if result and 'housing_ml' in result:
            ml_results = result['housing_ml']
            
            if ml_results.get('status') == 'success':
                display_results(ml_results, logger)
                save_results(ml_results, logger)
            else:
                logger.error(f"ML analysis failed: {ml_results.get('message', 'Unknown error')}")
        else:
            logger.error("No ML results found in workflow output")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return 1
        
    return 0

def display_results(ml_results, logger):
    """Display the results in a formatted way."""
    logger.info("\n" + "="*50)
    logger.info("HOUSING PREDICTION RESULTS")
    logger.info("="*50)
    
    summary = ml_results.get('summary', {})
    model_performance = ml_results.get('model_performance', {})
    
    # Model Performance
    logger.info(f"\n📊 MODEL PERFORMANCE:")
    logger.info(f"   Best Model: {model_performance.get('best_model', 'Unknown')}")
    logger.info(f"   ROC AUC Score: {model_performance.get('roc_auc', 0):.3f}")
    logger.info(f"   Accuracy: {model_performance.get('accuracy', 0):.3f}")
    logger.info(f"   F1 Score: {model_performance.get('f1_score', 0):.3f}")
    
    # Prediction Summary
    logger.info(f"\n🏠 PREDICTION SUMMARY:")
    logger.info(f"   Total Properties Analyzed: {summary.get('total_properties_analyzed', 0)}")
    logger.info(f"   High Probability Leads: {summary.get('high_probability_count', 0)}")
    logger.info(f"   Medium Probability: {summary.get('medium_probability_count', 0)}")
    logger.info(f"   Low Probability: {summary.get('low_probability_count', 0)}")
    logger.info(f"   Average Sell Probability: {summary.get('avg_sell_probability', 0):.3f}")
    
    # Top Features
    feature_importance = ml_results.get('feature_importance', {})
    logger.info(f"\n🔍 TOP PREDICTIVE FEATURES:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:8], 1):
        logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
    
    # High Priority Leads
    high_prob_leads = ml_results.get('high_probability_leads', [])
    logger.info(f"\n🎯 TOP HIGH-PROBABILITY LEADS:")
    for i, lead in enumerate(high_prob_leads[:5], 1):
        logger.info(f"   {i}. {lead.get('address', 'N/A')} - Probability: {lead.get('sell_probability', 0):.3f}")
    
    # Insights and Recommendations
    insights = ml_results.get('insights', {})
    recommendations = insights.get('recommendations', [])
    logger.info(f"\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec}")

def save_results(ml_results, logger):
    """Save results to files for further analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Save detailed predictions
        predictions = ml_results.get('predictions', [])
        if predictions:
            df_predictions = pd.DataFrame(predictions)
            predictions_file = f"housing_predictions_{timestamp}.csv"
            df_predictions.to_csv(predictions_file, index=False)
            logger.info(f"\n💾 Detailed predictions saved to: {predictions_file}")
        
        # Save high probability leads
        high_prob_leads = ml_results.get('high_probability_leads', [])
        if high_prob_leads:
            df_leads = pd.DataFrame(high_prob_leads)
            leads_file = f"high_probability_leads_{timestamp}.csv"
            df_leads.to_csv(leads_file, index=False)
            logger.info(f"💾 High probability leads saved to: {leads_file}")
        
        # Save complete results as JSON
        results_file = f"housing_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(ml_results, default=str))
            json.dump(serializable_results, f, indent=2)
        logger.info(f"💾 Complete analysis saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    sys.exit(main())
