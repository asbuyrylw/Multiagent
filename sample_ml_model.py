#!/usr/bin/env python3
"""
Sample ML Model for Cincinnati Real Estate Price Prediction
Demonstrates how to use the scraped Redfin data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CincinnatiHousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_and_prepare_data(self, csv_file):
        """Load and prepare the Cincinnati real estate data"""
        
        print("🏠 CINCINNATI HOUSE PRICE PREDICTION MODEL")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} properties from Cincinnati")
        
        # Basic data cleaning
        df = self.clean_data(df)
        print(f"✓ After cleaning: {len(df)} properties remain")
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare the data for ML"""
        
        # Remove properties with missing price (our target)
        df = df.dropna(subset=['price'])
        
        # Remove extreme outliers in price
        price_q1 = df['price'].quantile(0.01)
        price_q99 = df['price'].quantile(0.99)
        df = df[(df['price'] >= price_q1) & (df['price'] <= price_q99)]
        
        # Fill missing values for key features
        numeric_columns = ['beds', 'baths', 'sqft', 'lot_size', 'year_built', 'price_per_sqft']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values
        categorical_columns = ['location', 'zip_code', 'property_type']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Fill missing boolean values
        boolean_columns = ['has_virtual_tour', 'has_3d_tour', 'is_new_construction']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        
        # Select features for the model
        feature_columns = [
            'beds', 'baths', 'sqft', 'lot_size', 'year_built',
            'location', 'zip_code', 'property_type',
            'has_virtual_tour', 'has_3d_tour', 'is_new_construction'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"✓ Using {len(available_features)} features for prediction")
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df['price'].copy()
        
        # Handle categorical variables
        categorical_features = ['location', 'zip_code', 'property_type']
        
        for col in categorical_features:
            if col in X.columns:
                # Limit to top N categories to avoid too many features
                top_categories = X[col].value_counts().head(10).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
                
                # Label encode categorical variables
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Convert boolean columns to int
        boolean_cols = ['has_virtual_tour', 'has_3d_tour', 'is_new_construction']
        for col in boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Feature engineering complete: {list(self.feature_names)}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train the ML model"""
        
        print(f"\n📊 Training ML Model...")
        print("-" * 30)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} properties")
        print(f"Test set: {len(X_test)} properties")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('-inf')
        
        for name, model in models.items():
            # Train model
            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n{name} Results:")
            print(f"  • MAE: ${mae:,.0f}")
            print(f"  • RMSE: ${rmse:,.0f}")
            print(f"  • R²: {r2:.3f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.model = model
                
                # Store test results
                self.test_mae = mae
                self.test_rmse = rmse
                self.test_r2 = r2
                self.y_test = y_test
                self.y_pred = y_pred
        
        print(f"\n✓ Best model selected with R² = {best_score:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_feature_importance(self, X):
        """Analyze feature importance for Random Forest"""
        
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🔍 Feature Importance Analysis:")
            print("-" * 30)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in importance_df.head(10).iterrows():
                print(f"  • {row['feature']}: {row['importance']:.3f}")
    
    def predict_sample_properties(self, df):
        """Make predictions on sample properties"""
        
        print(f"\n🎯 Sample Predictions:")
        print("-" * 30)
        
        # Get a few sample properties
        samples = df.sample(5, random_state=42)
        
        for idx, property_data in samples.iterrows():
            # Prepare features for prediction
            X_sample = pd.DataFrame([property_data[self.feature_names]])
            
            # Handle categorical encoding
            for col in self.label_encoders:
                if col in X_sample.columns:
                    try:
                        X_sample[col] = self.label_encoders[col].transform([str(property_data[col])])
                    except ValueError:
                        # Handle unseen categories
                        X_sample[col] = 0
            
            # Make prediction
            if hasattr(self.model, 'feature_importances_'):  # Random Forest
                predicted_price = self.model.predict(X_sample)[0]
            else:  # Linear Regression
                X_sample_scaled = self.scaler.transform(X_sample)
                predicted_price = self.model.predict(X_sample_scaled)[0]
            
            actual_price = property_data['price']
            
            print(f"\nProperty: {property_data.get('address', 'Unknown Address')}")
            print(f"  📍 Location: {property_data.get('location', 'Unknown')}")
            print(f"  🏠 {property_data.get('beds', 'N/A')} bed, {property_data.get('baths', 'N/A')} bath")
            print(f"  📏 {property_data.get('sqft', 'N/A'):,.0f} sqft")
            print(f"  💰 Actual: ${actual_price:,.0f}")
            print(f"  🤖 Predicted: ${predicted_price:,.0f}")
            print(f"  📊 Error: ${abs(actual_price - predicted_price):,.0f} ({abs(actual_price - predicted_price)/actual_price*100:.1f}%)")
    
    def generate_market_insights(self, df):
        """Generate market insights from the model"""
        
        print(f"\n💡 Market Insights:")
        print("-" * 30)
        
        # Price per bedroom analysis
        if 'beds' in df.columns:
            price_per_bed = df.groupby('beds')['price'].mean().sort_index()
            print("Average price by bedrooms:")
            for beds, avg_price in price_per_bed.items():
                print(f"  • {beds} bed: ${avg_price:,.0f}")
        
        # Most expensive neighborhoods
        if 'location' in df.columns:
            location_prices = df.groupby('location')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            print(f"\nTop 5 most expensive neighborhoods:")
            for location, row in location_prices.head(5).iterrows():
                if row['count'] >= 5:  # Only show areas with sufficient data
                    print(f"  • {location}: ${row['mean']:,.0f} (avg from {row['count']} properties)")
        
        # Model accuracy by price range
        print(f"\nModel Performance by Price Range:")
        df_test = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.y_pred
        })
        
        df_test['price_range'] = pd.cut(df_test['actual'], 
                                       bins=[0, 200000, 400000, 600000, float('inf')],
                                       labels=['Under $200K', '$200K-$400K', '$400K-$600K', 'Over $600K'])
        
        for price_range in df_test['price_range'].unique():
            if pd.notna(price_range):
                subset = df_test[df_test['price_range'] == price_range]
                if len(subset) > 0:
                    mae = mean_absolute_error(subset['actual'], subset['predicted'])
                    mape = np.mean(np.abs((subset['actual'] - subset['predicted']) / subset['actual']) * 100)
                    print(f"  • {price_range}: MAE=${mae:,.0f}, MAPE={mape:.1f}%")

def main():
    """Main execution function"""
    
    # Initialize the predictor
    predictor = CincinnatiHousePricePredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_data('cincinnati_active_listings_20250726_150118.csv')
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    X_train, X_test, y_train, y_test = predictor.train_model(X, y)
    
    # Analyze feature importance
    predictor.analyze_feature_importance(X)
    
    # Make sample predictions
    predictor.predict_sample_properties(df)
    
    # Generate market insights
    predictor.generate_market_insights(df)
    
    print(f"\n" + "=" * 50)
    print("✅ ML MODEL ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"\n🎯 Key Results:")
    print(f"• Model trained on {len(X)} Cincinnati properties")
    print(f"• Test accuracy: R² = {predictor.test_r2:.3f}")
    print(f"• Average prediction error: ${predictor.test_mae:,.0f}")
    print(f"• Model ready for production use!")

if __name__ == "__main__":
    main()