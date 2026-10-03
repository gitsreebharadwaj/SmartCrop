import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def main():
    print("Loading historical agricultural data...")
    try:
        df = pd.read_csv('data/historical_crop_data.csv')
    except FileNotFoundError:
        print("Error: data/historical_crop_data.csv not found.")
        return

    required_columns = [
        'Region', 'Soil_Type', 'Temperature_C', 'Rainfall_mm', 'Humidity_Percent',
        'Crop', 'Area_Acres', 'Yield_Kg'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing.")
            return

    print("Preparing data...")
    df = df.dropna(subset=required_columns)

    X = df[['Region', 'Soil_Type', 'Temperature_C', 'Rainfall_mm', 'Humidity_Percent', 'Crop', 'Area_Acres']]
    y = df['Yield_Kg']

    categorical_features = ['Region', 'Soil_Type', 'Crop']
    numeric_features = ['Temperature_C', 'Rainfall_mm', 'Humidity_Percent', 'Area_Acres']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}\n")

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/yield_model.joblib')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
