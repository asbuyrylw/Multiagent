from agents.base_agent import BaseAgent
from tools.housing_ml_tool import prepare_housing_features, train_housing_prediction_model, predict_sales_likelihood, generate_model_insights
from utils.logger import setup_logger

class HousingMLAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.logger = setup_logger(f"housing_ml_agent_{name}")

    def handle(self, task):
        """
        Handle ML training and prediction tasks for housing data.
        """
        self.logger.info(f"HousingMLAgent handling task: {task}")
        
        try:
            property_data = task.get('property_data')
            census_data = task.get('census_data')
            
            if property_data is None or census_data is None:
                raise ValueError("Missing property_data or census_data in task")
            
            # Prepare features
            self.logger.info("Preparing features for ML model")
            prepared_data = prepare_housing_features(property_data, census_data)
            
            # Train models
            self.logger.info("Training housing prediction models")
            model_results, best_model_name, feature_columns = train_housing_prediction_model(prepared_data)
            
            # Make predictions on all data
            self.logger.info("Making predictions on all properties")
            best_model_info = model_results[best_model_name]
            predictions = predict_sales_likelihood(
                best_model_info, 
                prepared_data, 
                feature_columns, 
                best_model_name
            )
            
            # Generate insights
            self.logger.info("Generating model insights")
            insights = generate_model_insights(model_results, best_model_name, prepared_data)
            
            # Create summary results
            high_probability_properties = predictions[predictions['sell_probability'] > 0.7]
            
            result = {
                'status': 'success',
                'model_performance': {
                    'best_model': best_model_name,
                    'roc_auc': best_model_info['metrics']['roc_auc'],
                    'accuracy': best_model_info['metrics']['accuracy'],
                    'f1_score': best_model_info['metrics']['f1_score']
                },
                'predictions': predictions.to_dict('records'),
                'high_probability_leads': high_probability_properties.to_dict('records'),
                'insights': insights,
                'feature_importance': dict(list(best_model_info['feature_importance'].items())[:15]),
                'summary': {
                    'total_properties_analyzed': len(predictions),
                    'high_probability_count': len(high_probability_properties),
                    'medium_probability_count': len(predictions[predictions['confidence_level'] == 'Medium']),
                    'low_probability_count': len(predictions[predictions['confidence_level'] == 'Low']),
                    'avg_sell_probability': predictions['sell_probability'].mean(),
                    'model_accuracy': best_model_info['metrics']['accuracy']
                }
            }
            
            self.logger.info(f"ML analysis complete. Found {len(high_probability_properties)} high-probability leads")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            return {
                'status': 'error',
                'message': f"Failed to complete ML analysis: {str(e)}"
            }
