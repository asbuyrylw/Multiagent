
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def train_and_predict(task):
    # Handle both text_data format (from scraper) and CSV format
    if "text_data" in task:
        # Process text data from scraper
        text_data = task["text_data"]
        
        # For demo purposes, we'll create mock training data
        # In a real scenario, you'd have labeled training data
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Create mock labels for demo (in real scenario, you'd have actual labels)
        mock_labels = np.random.choice(['good_lead', 'poor_lead'], len(text_data))
        
        # Vectorize the text
        X = vectorizer.fit_transform(text_data).toarray()
        y = mock_labels
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        
        # Make predictions on all data
        predictions = model.predict(X)
        probs = model.predict_proba(X)
        
        pred_list = []
        for i in range(len(predictions)):
            pred_list.append({
                "input": f"Text sample {i+1}",
                "prediction": predictions[i],
                "confidence": max(probs[i])
            })
        
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        importances = dict(zip(feature_names, model.feature_importances_))
        
        return {
            "model": "RandomForestClassifier",
            "accuracy": acc,
            "predictions": pred_list,
            "feature_importance": dict(list(importances.items())[:10]),  # Top 10 features
            "text_samples_processed": len(text_data)
        }
    
    elif "training_data_path" in task:
        # Original CSV-based functionality
        train_df = pd.read_csv(task["training_data_path"])
        pred_df = pd.read_csv(task["prediction_data_path"])
        target = task["target_column"]

        X = train_df.drop(columns=[target])
        y = train_df[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))

        predictions = model.predict(pred_df)
        probs = model.predict_proba(pred_df)

        pred_list = []
        for i in range(len(predictions)):
            pred_list.append({
                "input": pred_df.iloc[i].to_dict(),
                "prediction": predictions[i],
                "confidence": max(probs[i])
            })

        importances = dict(zip(X.columns, model.feature_importances_))

        return {
            "model": "RandomForestClassifier",
            "accuracy": acc,
            "predictions": pred_list,
            "feature_importance": importances
        }
    
    else:
        raise ValueError("Task must contain either 'text_data' or 'training_data_path'")
