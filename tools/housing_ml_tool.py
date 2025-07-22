import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def prepare_housing_features(property_data, census_data):
    """
    Combine and engineer features from property and census data.
    """
    print("Preparing housing features for ML model...")
    
    # Convert property data to DataFrame if it's not already
    if not isinstance(property_data, pd.DataFrame):
        df = pd.DataFrame(property_data)
    else:
        df = property_data.copy()
    
    # Add census data as features (broadcast to all properties)
    for key, value in census_data.items():
        if key != 'location' and isinstance(value, (int, float)):
            df[f'census_{key}'] = value
    
    # Feature engineering
    df['price_per_sqft'] = df['current_value'] / df['sqft']
    df['equity_to_value_ratio'] = df['equity'] / df['current_value']
    df['years_since_purchase'] = df['days_since_last_sale'] / 365.25
    df['sales_frequency'] = df['past_sales_count'] / df['years_since_purchase'].clip(lower=1)
    df['value_growth_rate'] = df['value_appreciation'] / df['years_since_purchase'].clip(lower=1)
    
    # Property type features
    df['bedrooms_per_sqft'] = df['bedrooms'] / df['sqft'] * 1000
    df['bathrooms_per_bedroom'] = df['bathrooms'] / df['bedrooms']
    df['is_luxury'] = (df['current_value'] > df['current_value'].quantile(0.8)).astype(int)
    df['is_starter_home'] = (df['current_value'] < df['current_value'].quantile(0.3)).astype(int)
    
    # Market timing features
    df['market_appreciation_vs_area'] = df['value_appreciation'] - df['census_income_change_1yr']
    df['overvalued_vs_neighbors'] = (df['value_vs_neighbors'] > 0.1).astype(int)
    df['rapid_appreciation'] = (df['value_appreciation'] > 0.2).astype(int)
    
    # Financial pressure indicators
    df['high_mortgage_burden'] = (df['mortgage_balance'] / df['current_value'] > 0.7).astype(int)
    df['low_equity'] = (df['equity_ratio'] < 0.3).astype(int)
    df['recent_purchase'] = (df['days_since_last_sale'] < 730).astype(int)  # Less than 2 years
    
    # Area growth indicators
    df['growing_area'] = (df['census_population_change_1yr'] > 0.05).astype(int)
    df['economic_growth'] = (df['census_economic_health'] > 0.6).astype(int)
    df['high_mobility_area'] = (df['census_mobility_index'] > 0.25).astype(int)
    
    print(f"Feature engineering complete. Dataset shape: {df.shape}")
    return df

def train_housing_prediction_model(prepared_data, target_column='will_sell_within_year'):
    """
    Train multiple ML models to predict housing sales likelihood.
    """
    print("Training housing prediction models...")
    
    # Prepare features and target
    X = prepared_data.drop([target_column, 'property_id', 'address'], axis=1, errors='ignore')
    y = prepared_data[target_column]
    
    # Remove any remaining non-numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Fit the model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute coefficients
            feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            feature_importance = {}
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'scaler': scaler if name == 'Logistic Regression' else None
        }
        
        print(f"{name} - ROC AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1_score']:.3f}")
    
    # Select best model based on ROC AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['roc_auc'])
    best_model_info = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} (ROC AUC: {best_model_info['metrics']['roc_auc']:.3f})")
    
    return results, best_model_name, X.columns.tolist()

def predict_sales_likelihood(model_info, new_data, feature_columns, model_name):
    """
    Make predictions on new data using the trained model.
    """
    print("Making predictions on new properties...")
    
    model = model_info['model']
    scaler = model_info.get('scaler')
    
    # Prepare features
    X_new = new_data[feature_columns]
    
    if scaler:
        X_new = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    # Create results dataframe
    results_df = new_data[['property_id', 'address']].copy()
    results_df['predicted_will_sell'] = predictions
    results_df['sell_probability'] = probabilities
    results_df['confidence_level'] = pd.cut(probabilities, 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
    
    # Sort by probability
    results_df = results_df.sort_values('sell_probability', ascending=False)
    
    print(f"Predictions complete. Found {predictions.sum()} properties likely to sell.")
    
    return results_df

def generate_model_insights(results, best_model_name, prepared_data):
    """
    Generate insights and recommendations from the model.
    """
    print("Generating model insights...")
    
    best_model_info = results[best_model_name]
    feature_importance = best_model_info['feature_importance']
    
    insights = {
        'model_performance': {
            'best_model': best_model_name,
            'roc_auc': best_model_info['metrics']['roc_auc'],
            'accuracy': best_model_info['metrics']['accuracy'],
            'precision': best_model_info['metrics']['precision'],
            'recall': best_model_info['metrics']['recall']
        },
        'top_features': list(feature_importance.items())[:10],
        'data_insights': {
            'total_properties': len(prepared_data),
            'properties_likely_to_sell': prepared_data['will_sell_within_year'].sum(),
            'sell_rate': prepared_data['will_sell_within_year'].mean(),
            'avg_property_value': prepared_data['current_value'].mean(),
            'avg_equity_ratio': prepared_data['equity_ratio'].mean()
        }
    }
    
    # Generate recommendations
    recommendations = []
    
    top_feature = list(feature_importance.keys())[0]
    recommendations.append(f"Focus on properties with high {top_feature} - this is the strongest predictor")
    
    if 'equity_ratio' in feature_importance:
        recommendations.append("Target properties with high equity ratios - owners are more likely to sell")
    
    if 'days_since_last_sale' in feature_importance:
        recommendations.append("Consider properties that haven't sold recently - owners may be ready for a change")
    
    if any('census' in feature for feature in feature_importance.keys()):
        recommendations.append("Area demographics are important - focus on growing or changing neighborhoods")
    
    insights['recommendations'] = recommendations
    
    print("Model insights generated successfully.")
    return insights
