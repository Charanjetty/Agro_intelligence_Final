# Machine Learning Workflow for AgroIntelligence

This document explains the end-to-end process of building the crop recommendation system for AgroIntelligence, from defining the problem to deployment.

---

## 1. Problem Definition
**Goal:** To recommend the most suitable crops to farmers in Andhra Pradesh to maximize their yield and income.
**Success Metric:** Accuracy of crop prediction (currently using Top-3 accuracy).
**Constraints:** 
- **Data:** Must be specific to Andhra Pradesh districts.
- **Latency:** Predictions must be instant (real-time) on the web app.
- **Inputs:** Must use accessible data for farmers (District, Season, simple soil properties).

## 2. Data Collection & Storage
We created a custom dataset (`apcrop_dataset_realistic.csv`) because public datasets did not have the specific combination of Andhra Pradesh districts + recent weather data.

### How `apcrop_dataset_realistic.csv` was collected:
We used a Python script (`dataset_generation.ipynb`) to generate this data.
1.  **Locations:** We defined the 26 districts of Andhra Pradesh and their approximate latitude/longitude.
2.  **Weather Data (NASA POWER API):** 
    - The script connects to the **NASA POWER API** (a public source of satellite-based climate data).
    - For each district, it fetches 10 years (2015-2024) of monthly data:
        - **T2M:** Temperature at 2 meters.
        - **PRECTOT:** Precipitation (Rainfall).
    - It averages these values to get "Seasonal Rainfall" and "Average Temperature" for Kharif, Rabi, and Zaid seasons.
3.  **Soil Data:** We used "Deterministic Estimation". Since we didn't have 10,000 soil tests, we used government averages (e.g., "Guntur has predominantly Black Cotton Soil with high Nitrogen") to assign realistic soil values to each district.
4.  **Storage:** The final data is saved as a CSV file (`apcrop_dataset_realistic.csv`) which acts as our database for training.

## 3. Data Preprocessing
Before training, the raw data must be cleaned and formatted.
- **Handling Missing Values:** We use `KNNImputer` to fill in any missing numerical values (though our synthetic dataset is mostly clean).
- **Encoding Categorical Data:** Computers understand numbers, not words.
    - **One-Hot Encoding:** We convert text columns like `District` (e.g., "Guntur", "Krishna") and `Season` into binary columns (0s and 1s).
    - Example: `District_Guntur` = 1, `District_Krishna` = 0.

## 4. Feature Engineering
We select the most important features that affect crop growth:
- **Environmental:** `Avg_Temp_C`, `Seasonal_Rainfall_mm`, `Avg_Humidity_pct`
- **Soil:** `Soil_N_kg_ha` (Nitrogen), `Soil_P_kg_ha` (Phosphorus), `Soil_K_kg_ha` (Potassium), `Soil_pH`
- **Location:** `District`, `Mandal`
- **Time:** `Season`

## 5. Train/Validation/Test Split
We split our dataset to ensure the model can generalize to new data.
- **Training Set (80%):** Used to teach the model.
- **Test Set (20%):** Used to evaluate performance. The model *never* sees this data during training.

## 6. Model Selection & Training
**Algorithm Chosen:** **Random Forest Classifier**
- **Why?** It is robust, handles both numerical and categorical data well, and is less prone to overfitting than a single Decision Tree.
- **How it works:** It builds multiple "Decision Trees" (e.g., 100 trees). Each tree votes on the best crop. The forest takes the majority vote.
- **Training:** We run `model.fit(X_train, y_train)` where `X` is our features (soil, weather) and `y` is the target (Crop).

## 7. Evaluation & Optimization
We measure how well the model performs.
- **Metrics:** We look at **Accuracy** (percentage of correct predictions).
- **Top-3 Accuracy:** Since multiple crops can grow in the same place, we check if the *actual* best crop is in the model's top 3 recommendations.
- **Optimization:** We can tune "Hyperparameters" like the number of trees (`n_estimators`) or the maximum depth of the trees to improve results.

## 8. Deployment
The trained model is saved as a file (`crop_recommender_rf.joblib`).
- **Web App (Flask):** We built a web application using Flask (`app.py`).
- **Loading:** When the app starts, it loads the `.joblib` file into memory.
- **Prediction:** When a user clicks "Predict":
    1. The app takes user input (District, Season, etc.).
    2. It converts it into the same format as the training data.
    3. It passes it to the loaded model.
    4. The model returns the predicted crops.
    5. The app displays them to the user.

## 9. Monitoring & Maintenance
- **Tracking:** In a real-world scenario, we would log every prediction made to see what users are asking for.
- **Retraining:** As years pass, climate patterns change. We would need to re-run the `dataset_generation.ipynb` script to fetch new NASA weather data and re-train the model to keep it accurate.
