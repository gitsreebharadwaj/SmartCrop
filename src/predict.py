import pandas as pd
import joblib
import os

def normalize(series):
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min == 0:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)

def get_recommendation_and_reason(row, rank):
    if rank == 1:
        rec = "Recommended"
        reason = "Recommended because the crop has strong expected yield and profitability under the given conditions and has moderate regional concentration."
    elif rank <= 3:
        rec = "Suitable Alternative"
        reason = "Suitable alternative because the crop performs reasonably well but another crop has a higher overall score."
    else:
        rec = "Lower Priority"
        reason = "Lower priority because the crop has lower expected profitability and/or high regional concentration."
    return rec, reason

def main():
    print("Loading current conditions...")
    try:
        current_df = pd.read_csv('input/current_conditions.csv')
    except FileNotFoundError:
        print("Error: input/current_conditions.csv not found.")
        return

    try:
        hist_df = pd.read_csv('data/historical_crop_data.csv')
    except FileNotFoundError:
        print("Error: data/historical_crop_data.csv not found.")
        return

    try:
        model = joblib.load('models/yield_model.joblib')
    except FileNotFoundError:
        print("Error: models/yield_model.joblib not found. Run train_model.py first.")
        return

    print("Evaluating candidate crops...\n")

    # We need the single farmer's condition
    farmer_condition = current_df.iloc[0].to_dict()
    region = farmer_condition['Region']
    area = farmer_condition['Area_Acres']

    # Get unique crops
    candidate_crops = hist_df['Crop'].unique()

    # Pre-calculate economic averages by Region and Crop
    # Production cost is per acre
    hist_df['Cost_Per_Acre'] = hist_df['Production_Cost_INR'] / hist_df['Area_Acres']
    
    econ_agg = hist_df.groupby(['Region', 'Crop'])[['Selling_Price_INR_Per_Kg', 'Cost_Per_Acre']].mean().reset_index()
    
    # Pre-calculate regional share
    regional_data = hist_df[hist_df['Region'] == region]
    total_regional_area = regional_data['Area_Acres'].sum()
    crop_regional_area = regional_data.groupby('Crop')['Area_Acres'].sum().to_dict()

    results = []
    
    for crop in candidate_crops:
        # Construct features for prediction
        input_data = pd.DataFrame([{
            'Region': region,
            'Soil_Type': farmer_condition['Soil_Type'],
            'Temperature_C': farmer_condition['Temperature_C'],
            'Rainfall_mm': farmer_condition['Rainfall_mm'],
            'Humidity_Percent': farmer_condition['Humidity_Percent'],
            'Crop': crop,
            'Area_Acres': area
        }])

        predicted_yield = model.predict(input_data)[0]
        
        # Get economic info
        econ_info = econ_agg[(econ_agg['Region'] == region) & (econ_agg['Crop'] == crop)]
        if not econ_info.empty:
            avg_selling_price = econ_info['Selling_Price_INR_Per_Kg'].values[0]
            avg_cost_per_acre = econ_info['Cost_Per_Acre'].values[0]
        else:
            # Fallback to global average if regional data is missing for this crop
            avg_selling_price = hist_df[hist_df['Crop'] == crop]['Selling_Price_INR_Per_Kg'].mean()
            avg_cost_per_acre = hist_df[hist_df['Crop'] == crop]['Cost_Per_Acre'].mean()
            
        estimated_revenue = predicted_yield * avg_selling_price
        estimated_production_cost = avg_cost_per_acre * area
        estimated_profit = estimated_revenue - estimated_production_cost
        
        # Regional share
        crop_area = crop_regional_area.get(crop, 0.0)
        if total_regional_area > 0:
            regional_share = (crop_area / total_regional_area) * 100
        else:
            regional_share = 0.0
            
        results.append({
            'Crop': crop,
            'Predicted_Yield_Kg': round(predicted_yield, 2),
            'Estimated_Selling_Price_INR_Per_Kg': round(avg_selling_price, 2),
            'Estimated_Revenue_INR': round(estimated_revenue, 2),
            'Estimated_Production_Cost_INR': round(estimated_production_cost, 2),
            'Estimated_Profit_INR': round(estimated_profit, 2),
            'Regional_Crop_Share_Percent': round(regional_share, 2)
        })
        
    results_df = pd.DataFrame(results)
    
    # Calculate scores
    results_df['Yield_Score'] = normalize(results_df['Predicted_Yield_Kg'])
    results_df['Profit_Score'] = normalize(results_df['Estimated_Profit_INR'])
    
    # For regional balance, lower share = higher score (bounded between 0 and 1)
    max_share = results_df['Regional_Crop_Share_Percent'].max()
    if max_share > 0:
        results_df['Regional_Balance_Score'] = 1.0 - (results_df['Regional_Crop_Share_Percent'] / max_share)
    else:
        results_df['Regional_Balance_Score'] = 1.0
        
    results_df['Final_Score'] = (
        0.35 * results_df['Yield_Score'] + 
        0.45 * results_df['Profit_Score'] + 
        0.20 * results_df['Regional_Balance_Score']
    )
    
    # Sort and assign recommendations
    results_df = results_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    
    recommendations = []
    reasons = []
    for idx, row in results_df.iterrows():
        rank = idx + 1
        rec, reason = get_recommendation_and_reason(row, rank)
        recommendations.append(rec)
        reasons.append(reason)
        
    results_df['Recommendation'] = recommendations
    results_df['Reason'] = reasons

    print("Recommendations:\n")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['Crop']}")
        print(f"   Expected Yield: {row['Predicted_Yield_Kg']} Kg")
        print(f"   Expected Profit: {row['Estimated_Profit_INR']} INR")
        print(f"   Regional Share: {row['Regional_Crop_Share_Percent']}%\n")
        
    os.makedirs('output', exist_ok=True)
    results_df.to_csv('output/recommendations.csv', index=False)
    print("Recommendation file generated successfully.")

if __name__ == '__main__':
    main()
