# AgroIntelligence 2.0 - Complete Project Documentation

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Installation & Setup](#installation--setup)
6. [User Guide](#user-guide)
7. [API Reference](#api-reference)
8. [Database Schema](#database-schema)
9. [ML Model Details](#ml-model-details)
10. [Deployment](#deployment)

---

## Executive Summary

AgroIntelligence 2.0 is an AI-powered crop recommendation system designed specifically for farmers in Andhra Pradesh, India. The platform leverages machine learning to provide data-driven crop suggestions based on soil conditions, climate data, and regional agricultural patterns.

### Key Highlights
- **AI Accuracy**: Neural network model trained on 18,240+ data points
- **Coverage**: All 26 districts of Andhra Pradesh
- **Real-time Data**: Live weather integration via Open-Meteo API
- **User-Friendly**: Dual mode prediction (Manual & Auto)
- **Comprehensive**: Irrigation, fertilizer, weather, and scheme suggestions

---

## Features

### 🎯 Core Features

#### 1. **Dual Prediction Modes**
- **Manual Mode**: For farmers with soil test data
  - Input exact NPK values (Nitrogen, Phosphorus, Potassium)
  - Specify pH levels and organic carbon
  - Better precision for soil-tested farms
  
- **Auto Mode**: For quick recommendations
  - Select district only
  - System auto-fills soil and climate data
  - Uses district-level averages

#### 2. **Crop Suggestion Section** ⭐ NEW
When users receive their top 3 crop recommendations, they can click on any crop to view:

##### **Irrigation Suggestions**
- Water requirement (mm/season)
- Recommended irrigation method (drip/sprinkler/flood)
- Weekly watering schedule
- Critical growth stages

##### **Fertilizer Recommendations**
- NPK ratio specific to the crop
- Application schedule (basal, tillering, flowering)
- Split application timing
- Quantity in kg/hectare

##### **Live Weather Insights**
- Current temperature, wind speed, and conditions
- Real-time data from Open-Meteo
- Weather alerts for farming activities
- Best application days

##### **Government Schemes**
- YSR Rythu Bharosa (₹13,500/year)
- PM-KISAN (₹6,000/year)
- PM Fasal Bima Yojana (Crop Insurance)
- Micro Irrigation Fund
- Direct links to scheme portals

#### 3. **User Authentication**
- Secure signup/login system
- Email verification (simulated)
- User profile management
- Farm details tracking
- Prediction history

#### 4. **Weather Integration**
- Real-time weather data
- 12-hour forecasts
- Temperature, humidity, precipitation
- Wind speed and weather codes

#### 5. **AI Chatbot**
- Keyword-based question answering
- Common farming queries
- Soil, irrigation, fertilizer guidance
- Scheme information

#### 6. **Prediction History**
- Track all past predictions
- View historical recommendations
- Compare seasonal changes

---

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    User[👨‍🌾 Farmer/User]
    Web[🌐 Flask Web Application]
    Auth[🔐 Authentication Module]
    Pred[🧠 Prediction Engine]
    Data[📊 District Data Service]
    Weather[🌦️ Weather Service]
    Schemes[🏛️ Schemes Service]
    Chatbot[💬 Chatbot Service]
    DB[(🗄️ SQLite/PostgreSQL)]
    ML[🤖 TensorFlow Model]
    API[☁️ Open-Meteo API]
    
    User --> Web
    Web --> Auth
    Auth --> DB
    Web --> Pred
    Pred --> ML
    Pred --> Data
    Web --> Weather
    Weather --> API
    Web --> Schemes
    Web --> Chatbot
    DB --> Pred
```

### Module Breakdown

#### **Authentication Module**
- User registration and login
- Password hashing (Werkzeug)
- Session management (Flask-Login)
- Profile updates

#### **Prediction Module**
- Core ML engine
- Neural network (MLP) with TensorFlow
- Input preprocessing
- Top-3 crop recommendations

#### **Data Service Module**
- District-wise soil data
- Seasonal averages
- Historical agricultural data
- Crop-specific guidance

#### **Weather Module**
- Real-time API integration
- Current conditions
- Hourly forecasts
- District coordinate mapping

#### **Suggestion Module** ⭐ NEW
- Crop-specific detailed guidance
- Irrigation planning
- Fertilizer scheduling
- Weather-based recommendations
- Scheme filtering

---

## Technology Stack

### Backend
- **Framework**: Flask 3.0.0
- **Language**: Python 3.8+
- **ORM**: Flask-SQLAlchemy 3.1.1
- **Authentication**: Flask-Login 0.6.3
- **Server**: Gunicorn 21.2.0 (production)

### Machine Learning
- **Framework**: TensorFlow 2.15.0 (CPU)
- **Preprocessing**: Scikit-learn 1.3.0
- **Model Type**: Multi-Layer Perceptron (MLP)
- **Data Processing**: Pandas 2.0.3, NumPy 1.24.3

### Frontend
- **HTML/CSS**: Tailwind CSS (via CDN)
- **JavaScript**: Vanilla JS (ES6+)
- **Icons**: Font Awesome

### Database
- **Development**: SQLite
- **Production**: PostgreSQL (via psycopg2-binary)

### External APIs
- **Weather**: Open-Meteo API (https://api.open-meteo.com)

---

## Quick Start

The easiest way to run the project is using the automated startup scripts which handle the virtual environment and dependencies for you.

### On Windows (Recommended)
1.  Open PowerShell or Command Prompt in the project folder.
2.  Run the following command:
    ```powershell
    .\run.ps1
    ```
    *Alternatively, you can simply double-click `run.bat`.*

### Manual Setup
If you prefer to set up manually:
1.  Create a virtual environment: `python -m venv .venv`
2.  Activate it: `.\.venv\Scripts\activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the app: `python app.py`
Open your browser and navigate to: **http://localhost:5000**

---

## User Guide

### 1. Registration
1. Click "Sign Up" on the landing page
2. Enter email, username, password, full name, and phone
3. Submit the form
4. You'll be redirected to login

### 2. Making Predictions

#### **Manual Mode**
1. Log in to your account
2. Click "Dashboard" / "Prediction"
3. Select "Start Manual Mode"
4. Fill in your soil test values:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - pH Level
   - Temperature, Humidity, Rainfall (optional)
   - District (optional)
5. Click "Get Recommendations"

#### **Auto Mode**
1. Log in to your account
2. Click "Dashboard" / "Prediction"
3. Select "Start Auto Mode"
4. Select your district
5. Select season (or auto-detect)
6. Click "Auto-Fetch Data & Predict"

### 3. Viewing Crop Suggestions ⭐ NEW
After receiving your top 3 crop recommendations:

1. Click "View Suggestions" button on any crop card
2. You'll be taken to the detailed suggestion page showing:
   - **Irrigation**: Water needs, methods, schedule
   - **Fertilizer**: NPK ratios, application timing
   - **Weather**: Live weather data for your district
   - **Schemes**: Government subsidies and assistance
   - **Pro Tips**: Best practices for the crop

### 4. Checking Weather
1. Navigate to "Weather" page
2. Select your district
3. View current conditions and forecast

### 5. Exploring Schemes
1. Click "Schemes" in navigation
2. Browse available government schemes
3. Click "Learn More" to visit official portals

### 6. Prediction History
1. Go to "History" page
2. View all your past predictions
3. See crops, scores, dates, and modes

---

## API Reference

### Authentication Endpoints

#### `POST /signup`
Register a new user
- **Body**: `email`, `username`, `password`, `full_name`, `phone`
- **Response**: Redirect to login on success

#### `POST /login`
User login
- **Body**: `email`, `password`, `remember`
- **Response**: Redirect to dashboard on success

#### `GET /logout`
User logout
- **Auth**: Required
- **Response**: Redirect to home

### Prediction Endpoints

#### `POST /predict`
Generate crop recommendations
- **Auth**: Required
- **Body**:
  ```json
  {
    "mode": "manual" | "auto",
    "district": "Guntur",
    "season": "Kharif",
    "soil_n": 90,
    "soil_p": 42,
    "soil_k": 43,
    "soil_ph": 6.5
  }
  ```
- **Response**:
  ```json
  {
    "recommendations": [
      {"crop": "Paddy", "score": 0.95},
      {"crop": "Maize", "score": 0.78},
      {"crop": "Cotton", "score": 0.65}
    ],
    "location_details": {...},
    "guidance": {...},
    "weather": {...}
  }
  ```

#### `GET /suggestion/<crop_name>` ⭐ NEW
View detailed suggestions for a crop
- **Auth**: Required
- **URL Params**: `crop_name` (e.g., "Paddy")
- **Response**: HTML page with irrigation, fertilizer, weather, schemes

### Data Endpoints

#### `GET /get_district_names`
Fetch all district names
- **Response**: `["Guntur", "Krishna", ...]`

#### `GET /get_district_data/<district_name>`
Get district-specific data
- **Response**:
  ```json
  {
    "district": "Guntur",
    "soil_type": "Black",
    "soil_ph": 7.2,
    "mandals": ["Guntur_Mandal_1", ...]
  }
  ```

#### `GET /auto_defaults?district=<>&season=<>`
Get auto mode defaults
- **Query Params**: `district`, `season`
- **Response**: District-season specific averages

### Weather Endpoints

#### `GET /api/weather/<district>`
Fetch live weather data
- **Response**:
  ```json
  {
    "current": {
      "temperature": 28,
      "windspeed": 12,
      "weathercode": 1
    },
    "hourly": [...]
  }
  ```

### Schemes & Chatbot

#### `GET /schemes`
List all government schemes
- **Response**: Array of schemes with title, description, link

#### `POST /chat`
Ask chatbot a question
- **Body**: `{"message": "How do I raise soil pH?"}`
- **Response**: `{"response": "Apply agricultural lime..."}`

---

## Database Schema

### User Model
```python
class User:
    id: Integer (PK)
    email: String (unique)
    username: String (unique)
    password_hash: String
    full_name: String
    phone: String
    district: String
    email_verified: Boolean
    profile_image: String
    farm_size: Float
    primary_crops: String
    created_at: DateTime
    last_login: DateTime
```

### Prediction Model
```python
class Prediction:
    id: Integer (PK)
    user_id: Integer (FK -> User)
    district: String
    mandal: String
    season: String
    soil_type: String
    water_source: String
    mode: String
    soil_ph: Float
    organic_carbon: Float
    soil_n: Float
    soil_p: Float
    soil_k: Float
    top_crop: String
    top_crop_score: Float
    second_crop: String
    second_crop_score: Float
    third_crop: String
    third_crop_score: Float
    created_at: DateTime
```

### ContactMessage Model
```python
class ContactMessage:
    id: Integer (PK)
    name: String
    email: String
    phone: String
    subject: String
    message: Text
    status: String
    created_at: DateTime
```

---

## ML Model Details

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP) Neural Network
- **Framework**: TensorFlow/Keras
- **Input Features**: 
  - Soil parameters (N, P, K, pH, Organic Carbon)
  - Climate (Temperature, Rainfall, Humidity)
  - Location (District, Season, Soil Type, Water Source)
- **Output**: Probability distribution over 30+ crop classes
- **Accuracy**: 95%+ on validation set

### Training Data
- **Dataset**: `apcrop_dataset_realistic.csv`
- **Rows**: 18,240 samples
- **Districts**: 26 (all of Andhra Pradesh)
- **Seasons**: Kharif, Rabi, Zaid
- **Years**: 2015-2024

### Preprocessing
1. **Missing Value Imputation**: KNN Imputer (k=5)
2. **Categorical Encoding**: One-hot encoding for district, season, soil type
3. **Feature Scaling**: Normalized numerical features
4. **Class Balancing**: Stratified sampling

### Model Files
- `croprecommender_mlp.h5`: Neural network weights
- `croprecommender_mlp.npz`: Feature columns and class labels
- `apcrop_dataset_realistic.csv`: Training dataset

---

## Deployment

### Local Development
```bash
python app.py
```
Runs on http://localhost:5000 with debug mode

### Production Deployment

#### Option 1: Render
1. Create account on Render.com
2. Connect GitHub repository
3. Set environment variables:
   - `DATABASE_URL`: PostgreSQL URL
   - `SECRET_KEY`: Random secret key
4. Deploy from `main` branch

#### Option 2: Heroku
1. Create Heroku app
2. Add PostgreSQL addon
3. Push code:
   ```bash
   git push heroku main
   ```

#### Option 3: AWS EC2
1. Launch Ubuntu instance
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```
3. Clone repository
4. Set up Gunicorn + Nginx
5. Configure systemd service

### Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SECRET_KEY=your-secret-key-here
```

---

## Troubleshooting

### Common Issues

**Issue**: Model not loading
- **Solution**: Ensure `croprecommender_mlp.h5` and `.npz` files exist in root directory

**Issue**: Weather API not working
- **Solution**: Check internet connection, API may have rate limits

**Issue**: Database errors
- **Solution**: Run `db.create_all()` in Flask shell to initialize tables

---

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Contact & Support

- **Developer**: Charan
- **GitHub**: https://github.com/Charanjetty/AgroI2.0
- **Issues**: Report on GitHub Issues tab

---

## Acknowledgments

- **NASA POWER**: Climate data source
- **Open-Meteo**: Weather API
- **Government of Andhra Pradesh**: Agricultural data and schemes
- **Farmers**: For their valuable feedback

---

## Version History

### v2.0 (Current)
- ⭐ Added Crop Suggestion Section
- Irrigation, fertilizer, weather, schemes per crop
- Enhanced UI with Tailwind CSS
- Improved prediction accuracy

### v1.0
- Initial release
- Basic prediction system
- User authentication
- Weather integration

---

**Last Updated**: December 22, 2024
