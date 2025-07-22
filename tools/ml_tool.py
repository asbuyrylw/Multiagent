
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_predict(task):
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
