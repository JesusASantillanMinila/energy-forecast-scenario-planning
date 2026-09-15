import os
import json
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

def calculate_metrics(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE) and Percentage Bias."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero (though unlikely in energy consumption)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    bias = (np.sum(y_pred - y_true) / np.sum(y_true)) * 100
    
    return mape, bias

def select_best_predictors(df, candidate_predictors, val_days=30):
    """
    Performs greedy forward selection to find the best regressors based on MAPE.
    """
    # Split into train and validation (last `val_days` of historical data)
    train = df.iloc[:-val_days]
    val = df.iloc[-val_days:]

    best_predictors = []
    current_best_mape = float('inf')
    current_best_bias = float('inf')
    
    remaining_predictors = list(candidate_predictors)

    while remaining_predictors:
        best_feature_to_add = None
        best_step_mape = float('inf')
        best_step_bias = float('inf')

        for feature in remaining_predictors:
            test_predictors = best_predictors + [feature]

            # Initialize model with native US holidays
            m = Prophet()
            m.add_country_holidays(country_name='US')
            
            for p in test_predictors:
                m.add_regressor(p)

            # Fit and predict
            m.fit(train)
            forecast = m.predict(val[['ds'] + test_predictors])

            # Evaluate
            mape, bias = calculate_metrics(val['y'], forecast['yhat'])

            if mape < best_step_mape:
                best_step_mape = mape
                best_step_bias = bias
                best_feature_to_add = feature

        # If adding the best feature improved our overall MAPE, keep it
        if best_step_mape < current_best_mape:
            current_best_mape = best_step_mape
            current_best_bias = best_step_bias
            best_predictors.append(best_feature_to_add)
            remaining_predictors.remove(best_feature_to_add)
        else:
            # Stop searching if no remaining features improve the model
            break 

    return best_predictors, current_best_mape, current_best_bias

def forecast_prophet(df_final, output_dir='models'):
    """
    Main pipeline to train, evaluate, and save models for each city.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Only include predictors that make sense for energy demand
    candidate_predictors = [
        'Temp_Mean_C', 'Humidity_Pct', 'Wind_Speed_kmh',
        'Solar_Irradiance_MJm2', 'HDD', 'CDD', 'total_data_center_count'
    ]

    cities = df_final['City'].unique()

    for city in cities:
        print(f"\n--- Optimizing model for {city} ---")
        df_city = df_final[df_final['City'] == city].copy()

        # Prophet strictly requires 'ds' (datetime) and 'y' (target) columns
        df_city = df_city.rename(columns={'Date': 'ds', 'MWh_Consumed': 'y'})
        
        # Filter for historical data (where 'y' is not null)
        df_hist = df_city.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

        # 1. Select the best predictors
        best_preds, mape, bias = select_best_predictors(df_hist, candidate_predictors)
        print(f"Selected Predictors: {best_preds}")
        print(f"Validation MAPE: {mape:.2f}% | Validation Bias: {bias:.2f}%")

        # 2. Train final model on ALL historical data using the chosen predictors
        m_final = Prophet()
        m_final.add_country_holidays(country_name='US')
        for p in best_preds:
            m_final.add_regressor(p)

        m_final.fit(df_hist)

        # 3. Save model to JSON
        model_path = os.path.join(output_dir, f"prophet_model_{city}.json")
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(m_final), fout)
            
        print(f"Model saved to {model_path}")

        results[city] = {
            'best_predictors': best_preds,
            'mape': mape,
            'bias': bias,
            'model_path': model_path
        }

    return results