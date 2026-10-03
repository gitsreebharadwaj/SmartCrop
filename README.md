# SmartCrop

## 1. Project Overview
SmartCrop is an ML-based agricultural decision-support system. Given a region, soil conditions, and climatic conditions, it evaluates multiple crops and recommends suitable crops based on historical agricultural observations while preventing excessive regional concentration.

## 2. Problem Statement
Farmers often rely on traditional practices to select crops, potentially ignoring changing environmental conditions and market profitability. Additionally, if all farmers in a region select the most profitable crop, it leads to oversupply, soil degradation, and market crashes.

## 3. Objective
To build a system that evaluates candidate crops for a given set of conditions and recommends crops balancing expected yield, expected profitability, and regional crop diversity.

## 4. How SmartCrop Works
SmartCrop does **not** store farmer profiles or personal data. Instead, it maintains a historical dataset of agricultural observations. When a new farmer queries the system, they input their current soil, climate, and area. The system evaluates all candidate crops internally, predicts yield and economic performance, and scores each crop before recommending them.

## 5. Data Structure
- `data/historical_crop_data.csv`: Synthetic historical observations of Region, Soil, Climate, Crop, Yield, and Cost.
- `input/current_conditions.csv`: The current environmental conditions (without the crop specified).
- `output/recommendations.csv`: Generated output containing predictions and scores for all crops.

> **Note:** The included dataset is synthetic/demo data created for demonstrating the SmartCrop ML pipeline. It is not real agricultural data and should not be used for real-world farming decisions.

## 6. Machine Learning Approach
- **Task:** Supervised Learning (Regression)
- **Model:** Random Forest Regressor
- **Features:** Region, Soil_Type, Temperature_C, Rainfall_mm, Humidity_Percent, Crop, Area_Acres
- **Target:** Yield_Kg (Expected Yield)

## 7. Recommendation Logic
The ML model **only** predicts yield. Once expected yield is predicted for each candidate crop, the system calculates expected revenue and profit using historical averages. 
A composite score is calculated using:
- **Yield Score** (35%)
- **Profit Score** (45%)
- **Regional Balance Score** (20%)

## 8. Regional Crop Balance
To avoid monoculture and oversupply, SmartCrop calculates the historical or current concentration of each crop in the requested region. Crops that already occupy a large share of the region receive a lower Regional Balance Score.

## 9. Technology Stack
- Python
- Pandas (Data processing)
- Scikit-learn (Machine learning)
- Joblib (Model persistence)
- CSV (Data storage)

## 10. Project Structure
```text
SmartCrop/
│
├── data/
│   └── historical_crop_data.csv
│
├── input/
│   └── current_conditions.csv
│
├── output/
│   └── recommendations.csv
│
├── models/
│   └── yield_model.joblib
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Installation
Install the minimal required dependencies:
```bash
pip install -r requirements.txt
```

## 12. How to Run
1. Train the machine learning model:
```bash
python src/train_model.py
```
2. Generate recommendations for the current conditions:
```bash
python src/predict.py
```

## 13. Example Input
`input/current_conditions.csv`
```csv
Region,Soil_Type,Temperature_C,Rainfall_mm,Humidity_Percent,Area_Acres
Bhimavaram,Loamy,28,1100,75,2
```

## 14. Example Output
`output/recommendations.csv` contains crops ranked by `Final_Score`, highlighting:
- Estimated Yield & Profit
- Regional Crop Share
- Individual component scores
- Recommendation status (e.g., "Recommended", "Suitable Alternative") and Reason.

## 15. Limitations
- Uses synthetic demo data, not real-world agricultural research.
- Prices and costs are derived from historical static averages rather than real-time market APIs.
- The model assumes static relationships between features and yield.
- Does not account for pests, diseases, or fertilizer management.

## 16. Future Scope
- Automated IVRS data collection.
- Mobile/web farmer interface.
- Real agricultural datasets integration.
- Real-time weather and market API integration.
- Future price forecasting capabilities.
- Advanced Recommendation Models and Deep Learning.

## 17. Interview Preparation / Core Flow
**Core explanation:**
"The system stores historical agricultural observations rather than farmer profiles. For a new farmer's region, soil, and climatic conditions, it evaluates different crops. A Random Forest Regressor predicts the expected yield for each candidate crop. Historical crop-specific price and cost information is then used to estimate revenue and profit. Finally, a recommendation score combines expected yield, profitability, and regional crop distribution so that the system does not recommend the same crop to everyone."

**Key Concepts to explain:**
- **Why are farmer profiles not stored?** The model learns the relationship between environment/crop and yield, not individual farmer behavior.
- **Why is yield prediction regression?** Because yield is a continuous numeric value.
- **Why Random Forest?** Handles non-linear relationships well and manages mixed feature types effectively.
- **What is Data Leakage?** Using price/cost as input features to predict yield would leak post-harvest information into the prediction phase. That is why they are excluded from model features.
- **Why is highest profit not always recommended?** To prevent regional over-concentration (market crashes and soil depletion).
