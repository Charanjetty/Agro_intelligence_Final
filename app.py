from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_from_directory, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
try:
    from google import genai
except ImportError:
    import google.generativeai as genai # Fallback if installation stalls
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Import models
from models import db, User, Prediction, ContactMessage
from sqlalchemy import desc

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "apcrop_dataset_realistic_enhanced.csv"
MODEL_PATH = BASE_DIR / "models" / "final_hybrid_model.pkl"
META_PATH = BASE_DIR / "models" / "model_metadata.json"

# Simulated OTP Storage (In-memory for demo)
otp_storage = {}

# Initialize Flask App (MUST BE AT TOP)
app = Flask(__name__)

# Configuration
database_url = os.environ.get('DATABASE_URL')
if not database_url:
    database_url = 'sqlite:///agrointelligence.db'
    print("⚠️  DATABASE_URL not set. Using local SQLite database.")
else:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(days=365)

# File upload configuration
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Email configuration REMOVED - Using Mobile Auth
# app.config['MAIL_SERVER'] = ...

app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET')

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Google Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Client initialized in AIService

print(f"✅ Database Configured: {app.config['SQLALCHEMY_DATABASE_URI'].split('://')[0]}://...")

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
# mail = Mail(app) # Removed

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image_path, max_size=(400, 400)):
    """Resize uploaded image to max dimensions"""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
    except Exception as e:
        print(f"Error resizing image: {e}")

def generate_otp():
    """Generate a 4-digit OTP"""
    import random
    return str(random.randint(1000, 9999))



EXCLUDE_COLUMNS = [
    "Year",
    "Suitable_Crops",
    "Fertilizer_Plan",
    "Irrigation_Plan",
    "Market_Price_Index",
    "Previous_Crop",
]

DISTRICT_COORDINATES: Dict[str, Dict[str, float]] = {
    "Alluri Sitharama Raju": {"lat": 17.6, "lon": 81.9},
    "Anakapalli": {"lat": 17.6868, "lon": 83.0033},
    "Anantapur": {"lat": 14.6819, "lon": 77.6006},
    "Annamayya": {"lat": 13.95, "lon": 78.5},
    "Bapatla": {"lat": 15.8889, "lon": 80.4593},
    "Chittoor": {"lat": 13.2172, "lon": 79.1003},
    "East Godavari": {"lat": 17.321, "lon": 82.04},
    "Eluru": {"lat": 16.7107, "lon": 81.0952},
    "Guntur": {"lat": 16.3067, "lon": 80.4365},
    "Kakinada": {"lat": 16.9891, "lon": 82.2475},
    "Konaseema": {"lat": 16.65, "lon": 82.0},
    "Krishna": {"lat": 16.57, "lon": 80.36},
    "Kurnool": {"lat": 15.8281, "lon": 78.0373},
    "NTR": {"lat": 16.5062, "lon": 80.648},
    "Nandyal": {"lat": 15.4888, "lon": 78.4836},
    "Palnadu": {"lat": 16.1167, "lon": 80.1667},
    "Parvathipuram Manyam": {"lat": 18.8, "lon": 83.433},
    "Prakasam": {"lat": 15.5, "lon": 79.5},
    "Nellore": {"lat": 14.4426, "lon": 79.9865},
    "Sri Satya Sai": {"lat": 14.4, "lon": 77.8},
    "Srikakulam": {"lat": 18.2989, "lon": 83.8938},
    "Tirupati": {"lat": 13.6288, "lon": 79.4192},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Vizianagaram": {"lat": 18.113, "lon": 83.3956},
    "West Godavari": {"lat": 16.7107, "lon": 81.0972},
    "Kadapa": {"lat": 14.4673, "lon": 78.8242},
}

GOVERNMENT_SCHEMES = [
    {
        "title": "Annadata Sukhibhava – PM Kisan",
        "description": "Flagship scheme providing ₹20,000 per annum (₹6,000 PM-Kisan + ₹14,000 State) to all farmers.",
        "link": "https://apagrisnet.gov.in/",
        "image": "scheme_financial.png"
    },
    {
        "title": "PM Fasal Bima Yojana (PMFBY)",
        "description": "Reinstated central-state crop insurance protecting farmers against unforeseen losses.",
        "link": "https://pmfby.gov.in/",
        "image": "scheme_insurance.png"
    },
    {
        "title": "PM-Kisan (Income Support)",
        "description": "Central scheme providing income support of ₹6,000 per year in three installments.",
        "link": "https://pmkisan.gov.in/",
        "image": "scheme_financial.png"
    },
    {
        "title": "Soil Health Card Scheme",
        "description": "Regular soil testing and issuance of cards to optimize fertilizer application.",
        "link": "https://soilhealth.dac.gov.in/",
        "image": "scheme_irrigation.png"
    },
    {
        "title": "PM Krishi Sinchayee Yojana (PMKSY)",
        "description": "Subsidy for drip/sprinkler irrigation and creation of water sources.",
        "link": "https://pmksy.gov.in/",
        "image": "scheme_irrigation.png"
    },
    {
        "title": "National Agriculture Market (eNAM)",
        "description": "Online trading platform for agricultural commodities ensuring better prices for farmers.",
        "link": "https://www.enam.gov.in/",
        "image": "scheme_financial.png"
    },
    {
        "title": "Kisan Credit Card (KCC)",
        "description": "Easy access to credit for farmers at subsidized interest rates for agricultural needs.",
        "link": "https://www.myscheme.gov.in/schemes/kisan-credit-card",
        "image": "scheme_financial.png"
    },
    {
        "title": "PM Kisan Maandhan (Pension)",
        "description": "Pension scheme for small and marginal farmers providing ₹3,000 monthly after age 60.",
        "link": "https://maandhan.in/",
        "image": "scheme_financial.png"
    },
    {
        "title": "AP Agri Services Portal",
        "description": "Official Andhra Pradesh agriculture services hub for crop booking, insurance, and more.",
        "link": "https://apagrisnet.gov.in/",
        "image": "scheme_financial.png"
    },
    {
        "title": "Vaddi Leni Runalu",
        "description": "Zero interest crop loans up to ₹1 lakh for farmers who repay within the stipulated time.",
        "link": "https://apagrisnet.gov.in/",
        "image": "scheme_financial.png"
    }
]

CHATBOT_KNOWLEDGE = [
    {
        "question": "How do I raise soil pH?",
        "keywords": ["ph", "soil", "acidic", "lime"],
        "answer": "Apply agricultural lime (2-3 t/ha) and incorporate organic matter like compost to buffer acidity. Re-test soil after one season.",
    },
    {
        "question": "How can I reduce irrigation costs?",
        "keywords": ["irrigation", "water", "drip", "sprinkler"],
        "answer": "Shift to drip/sprinkler systems, irrigate during cooler hours, and use mulching to reduce evaporation.",
    },
    {
        "question": "Which fertilizer is best for paddy?",
        "keywords": ["paddy", "fertilizer", "npk"],
        "answer": "A balanced plan is 100-120 kg N, 40-50 kg P2O5, 40-50 kg K2O per hectare split across basal, tillering, and panicle initiation stages.",
    },
    {
        "question": "How do I access government schemes?",
        "keywords": ["scheme", "government", "subsidy"],
        "answer": "Visit your nearest Rythu Bharosa Kendram (RBK) or apply online via the Annadata Unnathi platform.",
    },
]

def safe_mode(series: pd.Series) -> Optional[Any]:
    mode_values = series.mode()
    return mode_values.iloc[0] if not mode_values.empty else None

def safe_mean(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    value = float(series.mean())
    if pd.isna(value):
        return None
    return round(value, 2)

class DistrictDataService:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.copy()
        self.district_summary = self._build_district_summary()
        self.seasonal_summary = self._build_seasonal_summary()
        self.mandal_lookup = self._build_mandal_lookup()

    def _build_district_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for district, group in self.dataset.groupby("District"):
            summary[district] = self._summarize_group(group)
        return summary

    def _build_seasonal_summary(self) -> Dict[str, Dict[str, Any]]:
        seasonal: Dict[str, Dict[str, Any]] = {}
        for (district, season), group in self.dataset.groupby(["District", "Season"]):
            seasonal_key = f"{district}::{season}"
            seasonal[seasonal_key] = self._summarize_group(group)
        return seasonal

    def _build_mandal_lookup(self) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        for district, group in self.dataset.groupby("District"):
            lookup[district] = sorted(group["Mandal"].dropna().unique())
        return lookup

    def _summarize_group(self, group: pd.DataFrame) -> Dict[str, Any]:
        summary = {
            "district": group["District"].iloc[0],
            "mandal": safe_mode(group["Mandal"]),
            "season": safe_mode(group["Season"]),
            "soil_type": safe_mode(group["Soil_Type"]),
            "water_source": safe_mode(group["Water_Source"]),
            "secondary_crop": safe_mode(group["Secondary_Crop"]),
            "primary_crop": safe_mode(group["Primary_Crop"]),
            "soil_ph": safe_mean(group["Soil_pH"]),
            "organic_carbon": safe_mean(group["Organic_Carbon_pct"]),
            "soil_n": safe_mean(group["Soil_N_kg_ha"]),
            "soil_p": safe_mean(group["Soil_P_kg_ha"]),
            "soil_k": safe_mean(group["Soil_K_kg_ha"]),
            "rainfall": safe_mean(group["Seasonal_Rainfall_mm"]),
            "humidity": safe_mean(group["Avg_Humidity_pct"]),
            "temperature": safe_mean(group["Avg_Temp_C"]),
        }
        return summary

    def get_districts(self) -> List[str]:
        return sorted(self.district_summary.keys())

    def get_district_data(self, district: str) -> Dict[str, Any]:
        data = self.district_summary.get(district)
        if not data:
            raise ValueError(f"District '{district}' is not in the dataset.")
        return {**data, "mandals": self.mandal_lookup.get(district, [])}

    def get_auto_defaults(self, district: str, season: Optional[str]) -> Dict[str, Any]:
        if district not in self.district_summary:
            raise ValueError(f"District '{district}' is not in the dataset.")
        if season:
            seasonal_key = f"{district}::{season}"
            if seasonal_key in self.seasonal_summary:
                return self.seasonal_summary[seasonal_key]
        return self.district_summary[district]

    def build_model_payload(
        self,
        district: str,
        season: Optional[str],
        raw_payload: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        summary = self.get_auto_defaults(district, season)
        mandal = raw_payload.get("mandal") or summary.get("mandal")
        soil_type = raw_payload.get("soil_type") or summary.get("soil_type")
        water_source = raw_payload.get("water_source") or summary.get("water_source")
        season_value = season or raw_payload.get("season") or summary.get("season")

        def value_or_default(key: str, override_key: str) -> Optional[float]:
            if mode == "manual":
                override_value = raw_payload.get(override_key)
                if override_value in ("", None):
                    return summary.get(key)
                try:
                    return float(override_value)
                except (TypeError, ValueError):
                    return summary.get(key)
            return summary.get(key)

        return {
            "District": district,
            "Mandal": mandal,
            "Season": season_value,
            "Soil_Type": soil_type,
            "Soil_pH": value_or_default("soil_ph", "soil_ph"),
            "Organic_Carbon_pct": value_or_default("organic_carbon", "organic_carbon"),
            "Soil_N_kg_ha": value_or_default("soil_n", "soil_n"),
            "Soil_P_kg_ha": value_or_default("soil_p", "soil_p"),
            "Soil_K_kg_ha": value_or_default("soil_k", "soil_k"),
            "Avg_Temp_C": summary.get("temperature"),
            "Seasonal_Rainfall_mm": summary.get("rainfall"),
            "Avg_Humidity_pct": summary.get("humidity"),
            "Water_Source": water_source,
            "Secondary_Crop": summary.get("secondary_crop"),
            "Primary_Crop": summary.get("primary_crop"),
        }

    def fetch_guidance(self, district: str, crop: str) -> Dict[str, Any]:
        filtered = self.dataset[
            (self.dataset["District"] == district) & (self.dataset["Primary_Crop"] == crop)
        ]
        if filtered.empty:
            filtered = self.dataset[self.dataset["Primary_Crop"] == crop]
        if filtered.empty:
            return {}
        row = filtered.iloc[0]
        fertilizer_plan = self._safe_json(row.get("Fertilizer_Plan"))
        irrigation_plan = self._safe_json(row.get("Irrigation_Plan"))
        market_index = row.get("Market_Price_Index")
        return {
            "fertilizer_plan": fertilizer_plan,
            "irrigation_plan": irrigation_plan,
            "market_index": market_index,
        }

    @staticmethod
    def _safe_json(value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

class WeatherService:
    def __init__(self, coordinates: Dict[str, Dict[str, float]]) -> None:
        self.coordinates = coordinates

    def get_weather(self, district: str) -> Optional[Dict[str, Any]]:
        coords = self.coordinates.get(district)
        if not coords:
            return None
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current_weather": True,
            "hourly": "temperature_2m,relativehumidity_2m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto"
        }
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current_weather", {})
            hourly_payload = payload.get("hourly", {})
            daily_payload = payload.get("daily", {})
            
            # Process Hourly (Keep existing logic but limited)
            hourly_records: List[Dict[str, Any]] = []
            if hourly_payload:
                times = hourly_payload.get("time", []) or []
                temps = hourly_payload.get("temperature_2m", []) or []
                humidity = hourly_payload.get("relativehumidity_2m", []) or []
                precipitation = hourly_payload.get("precipitation", []) or []
                for idx, time_value in enumerate(times[:24]): # First 24 hours
                    hourly_records.append({
                        "time": time_value,
                        "temperature": self._safe_index(temps, idx),
                        "humidity": self._safe_index(humidity, idx),
                        "precipitation": self._safe_index(precipitation, idx),
                    })

            # Process Daily (New 7-day forecast)
            daily_records: List[Dict[str, Any]] = []
            if daily_payload:
                d_times = daily_payload.get("time", []) or []
                d_max_temps = daily_payload.get("temperature_2m_max", []) or []
                d_min_temps = daily_payload.get("temperature_2m_min", []) or []
                d_rain_sum = daily_payload.get("precipitation_sum", []) or []
                d_codes = daily_payload.get("weathercode", []) or []
                
                for idx, time_value in enumerate(d_times):
                    daily_records.append({
                        "date": time_value,
                        "max_temp": self._safe_index(d_max_temps, idx),
                        "min_temp": self._safe_index(d_min_temps, idx),
                        "rain_sum": self._safe_index(d_rain_sum, idx),
                        "weathercode": self._safe_index(d_codes, idx),
                    })

            # Generate Crop Advice
            advice = []
            if daily_records:
                total_rain = sum(d.get('rain_sum', 0) or 0 for d in daily_records[:3])
                avg_temp = sum(d.get('max_temp', 0) or 0 for d in daily_records[:3]) / 3
                
                if total_rain > 20:
                    advice.append("Heavy rain expected. Ensure proper drainage in fields.")
                if total_rain < 2:
                    advice.append("Dry spell ahead. Plan for irrigation.")
                if avg_temp > 35:
                    advice.append("Heatwave alert. Mulch soil to conserve moisture.")
                if not advice:
                    advice.append("Weather looks stable. Good time for fertilizer application.")

            return {
                "current": {
                    "temperature": current.get("temperature"),
                    "windspeed": current.get("windspeed"),
                    "weathercode": current.get("weathercode"),
                    "time": current.get("time"),
                },
                "hourly": hourly_records,
                "daily": daily_records,
                "advice": advice
            }
        except requests.RequestException:
            return None

    @staticmethod
    def _safe_index(values: List[Any], index: int) -> Optional[Any]:
        try:
            return values[index]
        except (IndexError, TypeError):
            return None

class SchemeService:
    def __init__(self, schemes: List[Dict[str, Any]]) -> None:
        self.schemes = schemes

    def list_schemes(self) -> List[Dict[str, Any]]:
        return self.schemes

class ChatbotService:
    def __init__(self, knowledge_base: List[Dict[str, str]]) -> None:
        self.knowledge_base = knowledge_base
        self.vectorizer = TfidfVectorizer()
        # Pre-train vectorizer on questions
        self.vectorizer.fit([k["question"] for k in self.knowledge_base])
        self.vectors = self.vectorizer.transform([k["question"] for k in self.knowledge_base])
        
        # Basic conversational patterns (English & Telugu)
        self.conversational_patterns = {
            r"\b(hi|hello|hey|greetings|namaste|namaskaram)\b": "Hello! Namaskaram! 🙏 I am your AgroIntelligence assistant. How can I help you? (నేను మీకు ఎలా సహాయపడగలను?)",
            r"\b(bye|goodbye|see you|ika vastanu)\b": "Goodbye! Happy farming! (శుభం!)",
            r"\b(thank|thanks|dhanyavadamulu)\b": "You're welcome! (ధన్యవాదాలు!) Let me know if you need anything else.",
            r"\b(who are you|what are you)\b": "I am an AI-powered farming assistant for Andhra Pradesh. (నేను మీ వ్యవసాయ సహాయకుడిని)",
            r"\b(help|sahayam)\b": "I can assist you with crop suggestions, weather info, and government schemes. Just ask! (పంటలు, వాతావరణం, పథకాల గురించి అడగండి)"
        }

    def answer(self, user_query: str) -> str:
        # 1. Check basic conversation first
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        for pattern, response in self.conversational_patterns.items():
            if re.search(pattern, user_query, re.IGNORECASE):
                return response

        # 2. Knowledge Base Search
        query_vec = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0.3:  # Threshold
            return self.knowledge_base[best_idx]["answer"]
        
        # 3. Fallback
        return (
            "I'm not sure about that. I specialize in farming advice for Andhra Pradesh. "
            "Could you try asking about crops, soil, or weather?"
        )

class AIService:
    def __init__(self):
        self.openai_client = None
        if OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                print(f"OpenAI Client Init Error: {e}")

        self.gemini_client = None
        self.gemini_model = None
        if GEMINI_API_KEY:
            try:
                # Compatibility check for new genai.Client vs old configure()
                if hasattr(genai, 'Client'):
                    self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                else:
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Gemini Client Init Error: {e}")

    def generate_content(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Unified generator with fallback: OpenAI -> Gemini -> Mock."""
        # 1. Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an agricultural expert. Provide advice in exactly 3 bullet points per section requested."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI Failed: {e}")

        # 2. Try Gemini
        if self.gemini_client:
            try:
                # New SDK Client usage
                response = self.gemini_client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                print(f"Gemini Client Failed: {e}")
        elif hasattr(self, 'gemini_model') and self.gemini_model:
            try:
                # Old SDK usage fallback
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini Model Failed: {e}")

        # 3. Fallback to Dynamic Mock
        return "⚠️ [Using Internal Knowledge] \n\n" + self._get_mock_response(prompt, context)

    def get_comprehensive_advice(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Consolidated method to get all advice in ONE request."""
        crop = context.get('crop', 'Crop')
        district = context.get('district', 'District')
        n, p, k = context.get('n', 0), context.get('p', 0), context.get('k', 0)
        ph = context.get('ph', 7.0)
        soil = context.get('soil_type', 'Red')
        water = context.get('water_source', 'Well')
        season = context.get('season', 'Kharif')
        weather = context.get('weather', 'No weather data')

        schemes_list = context.get('relevant_schemes', [])
        schemes_str = ", ".join(schemes_list) if schemes_list else "All available agriculture schemes"

        prompt = (
            f"Provide a comprehensive agricultural plan for {crop} in {district} district details:\n"
            f"- Soil: N={n}, P={p}, K={k}, pH={ph}, Type={soil}\n"
            f"- Environment: Season={season}, Water Source={water}, Weather Summary={weather}\n"
            f"- Available Relevant Schemes: {schemes_str}\n\n"
            "Produce EXACTLY 4 sections with EXACTLY 3 BULLET POINTS EACH:\n"
            "1. FERTILIZER PLAN: Specific doses based on NPK. Mention relevant subsidies (e.g., Annadata Unnathi) if applicable.\n"
            "2. IRRIGATION PLAN: Watering frequency and methods. Mention Annadata Sukhibhava or PM irrigation schemes if relevant.\n"
            "3. MARKET POTENTIAL: Price per kg and quintal, trends. Mention e-NAM or relevant market platforms if applicable.\n"
            "4. CLIMATE PRECAUTIONS: Weather-aware steps. Mention Crop Insurance (PMFBY) if relevant.\n\n"
            "Use clear section headers (e.g., ### FERTILIZER PLAN)."
        )

        response_text = self.generate_content(prompt, context)
        
        # Robust regex-based splitting
        import re
        sections = {
            "fertilizer": "",
            "irrigation": "",
            "market": "",
            "climate": ""
        }
        
        # Pattern to find sections starting with ### or 1. etc.
        patterns = {
            "fertilizer": [r"(?i)(?:###|1\.)\s*FERTILIZER.*?(?=\n\s*(?:###|2\.|3\.|4\.)|\Z)"],
            "irrigation": [r"(?i)(?:###|2\.)\s*IRRIGATION.*?(?=\n\s*(?:###|3\.|4\.)|\Z)"],
            "market": [r"(?i)(?:###|3\.)\s*MARKET.*?(?=\n\s*(?:###|4\.)|\Z)"],
            "climate": [r"(?i)(?:###|4\.)\s*CLIMATE.*?(?=\n\s*(?:###)|\Z)"]
        }

        for key, p_list in patterns.items():
            for p in p_list:
                match = re.search(p, response_text, re.DOTALL)
                if match:
                    # Clean up the prefix and headers
                    clean_text = match.group(0).strip()
                    # Remove common headers like ### FERTILIZER PLAN or 1. FERTILIZER
                    clean_text = re.sub(r'(?i)^(?:###|\d\.)\s*(?:FERTILIZER|IRRIGATION|MARKET|CLIMATE).*?\n', '', clean_text).strip()
                    sections[key] = clean_text
                    break

        # Emergency fallback if regex fails completely - better than dumping everything
        if not any(sections.values()):
            parts = re.split(r'\n\s*(?=###|\d\.)', response_text)
            for part in parts:
                p_up = part.upper()
                if "FERTILIZER" in p_up: sections["fertilizer"] = part
                elif "IRRIGATION" in p_up: sections["irrigation"] = part
                elif "MARKET" in p_up: sections["market"] = part
                elif "CLIMATE" in p_up: sections["climate"] = part
        
        # If still empty, use bits of the response text to avoid being totally blank
        if not sections["fertilizer"] and "⚠️" in response_text:
             # If it's the mock response, we can safely split it by \n\n
             mock_parts = response_text.split("\n\n")
             for part in mock_parts:
                 if "FERTILIZER" in part.upper(): sections["fertilizer"] = part
                 if "IRRIGATION" in part.upper(): sections["irrigation"] = part
                 if "MARKET" in part.upper(): sections["market"] = part
                 if "CLIMATE" in part.upper(): sections["climate"] = part

        return sections

    def _get_mock_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generates realistic mock data based on keywords and context."""
        prompt_lower = prompt.lower()
        ctx = context or {}
        crop = ctx.get("crop", "the crop")
        district = ctx.get("district", "your region")

        # Handle comprehensive prompt fallback
        if "comprehensive" in prompt_lower:
            return (
                self._get_mock_response("fertilizer", ctx) + "\n\n" +
                self._get_mock_response("irrigation", ctx) + "\n\n" +
                self._get_mock_response("market", ctx) + "\n\n" +
                self._get_mock_response("climate", ctx)
            )
        
        if "market" in prompt_lower:
            import random
            base_q = random.randint(2200, 3500)
            base_kg = round(base_q / 100, 2)
            return (
                f"### MARKET POTENTIAL\n"
                f"- Current price in {district}: ₹{base_q}/quintal (₹{base_kg}/kg).\n"
                f"- Market demand for {crop} remains stable with moderate growth.\n"
                f"- Consider selling in local mandis for immediate cash flow."
            )
        elif "fertilizer" in prompt_lower:
            n, p, k = ctx.get("n", 0), ctx.get("p", 0), ctx.get("k", 0)
            advice = [f"- Apply 45kg Urea as basal dose based on N={n}."]
            if p < 50: advice.append(f"- Your Phosphorus is low ({p}); add 20kg DAP.")
            else: advice.append("- Soil Phosphorus is adequate; maintain current balance.")
            advice.append("- Ensure even distribution near root zones for maximum uptake.")
            return "### FERTILIZER PLAN\n" + "\n".join(advice[:3])
            
        elif "irrigation" in prompt_lower:
            soil = ctx.get("soil_type", "Red")
            freq = "3-5 days" if soil == "Red" else "7-10 days"
            return (
                f"### IRRIGATION PLAN\n"
                f"- Maintain {freq} watering interval for your {soil} soil type.\n"
                f"- Use drip irrigation if available to conserve water.\n"
                f"- Avoid over-watering during maturity to prevent root rot."
            )
        elif "climate" in prompt_lower or "weather" in prompt_lower:
            return (
                f"### CLIMATE PRECAUTIONS\n"
                f"- Monitor {crop} for fungal indicators due to humidity.\n"
                f"- Clear drainage channels to prevent waterlogging during rains.\n"
                f"- Delay chemical sprays if local wind speed exceeds 12 km/h."
            )
        else:
            return "### ADVICE\n- Monitor crop health daily.\n- Keep field weed-free.\n- Consult official manuals."

    # Keep old methods as redirects to the new consolidated logic or for minor compatibility
    def get_market_potential(self, crop: str, district: str) -> str:
        res = self.get_comprehensive_advice({"crop": crop, "district": district})
        return res["market"]

    def get_fertilizer_plan(self, crop: str, n: float, p: float, k: float, ph: float) -> str:
        res = self.get_comprehensive_advice({"crop": crop, "n": n, "p": p, "k": k, "ph": ph})
        return res["fertilizer"]

    def get_irrigation_plan(self, crop: str, season: str, soil_type: str, water_source: str) -> str:
        res = self.get_comprehensive_advice({"crop": crop, "season": season, "soil_type": soil_type, "water_source": water_source})
        return res["irrigation"]

    def get_climate_suggestion(self, crop: str, duration: str, weather_context: str) -> str:
        res = self.get_comprehensive_advice({"crop": crop, "duration": duration, "weather": weather_context})
        return res["climate"]

class CropRecommendationEngine:
    def __init__(self, dataset: pd.DataFrame) -> None:
        # Lazy load model components
        self.model = None
        self.dataset = dataset.copy()
        # Pre-load dataset metadata (lightweight)
        if "Primary_Crop" not in self.dataset.columns:
            # Fallback if dataset is not loaded correctly
            self.placeholder_primary = "Paddy"
        else:
            self.placeholder_primary = safe_mode(self.dataset["Primary_Crop"]) or "Paddy"
            
        # We will initialize the rest in _ensure_loaded()

    def _ensure_loaded(self):
        """Loads the heavy model only when needed."""
        if self.model is not None:
            return

        print("⏳ Loading Hybrid Model...")
        self.model = joblib.load(MODEL_PATH)
        
        # Load classes
        if hasattr(self.model, "classes_"):
            self.classes = self.model.classes_
        else:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            self.classes = meta.get("classes", [])
            
        # HARDCODED: Exact 39 features expected by the model
        self.model_features = [
            "Soil_N_kg_ha", "Soil_P_kg_ha", "Soil_K_kg_ha", "Soil_pH", "Organic_Carbon_pct",
            "Avg_Temperature_C", "Rainfall_mm", "Humidity_pct",
            "District_Anakapalli", "District_Anantapur", "District_Annamayya", "District_Bapatla",
            "District_Chittoor", "District_East Godavari", "District_Eluru", "District_Guntur",
            "District_Kadapa", "District_Kakinada", "District_Konaseema", "District_Krishna",
            "District_Kurnool", "District_NTR", "District_Nandyal", "District_Nellore",
            "District_Palnadu", "District_Parvathipuram Manyam", "District_Prakasam",
            "District_Sri Satya Sai", "District_Srikakulam", "District_Tirupati",
            "District_Visakhapatnam", "District_Vizianagaram", "District_West Godavari",
            "Season_Rabi", "Season_Zaid",
            "Soil_Type_Black", "Soil_Type_Coastal Sands", "Soil_Type_Laterite", "Soil_Type_Red"
        ]

        print(f"✅ Model Loaded. Classes: {len(self.classes)} features: {len(self.model_features)}")

    def _build_feature_vector(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Constructs the exact 39-feature DataFrame expected by the model."""
        
        # District name normalization map (dataset name -> model feature name)
        district_map = {
            "Kadapa": "Kadapa",
            "Sri Satya Sai": "Sri Satya Sai",
            # Add other mappings if needed
        }
        
        # 1. Extract Numeric Values with correct mappings
        # Note: Payload uses capitalized keys from build_model_payload
        data = {
            "Soil_N_kg_ha": float(payload.get("Soil_N_kg_ha", 0)),
            "Soil_P_kg_ha": float(payload.get("Soil_P_kg_ha", 0)),
            "Soil_K_kg_ha": float(payload.get("Soil_K_kg_ha", 0)),
            "Soil_pH": float(payload.get("Soil_pH", 7.0)),
            "Organic_Carbon_pct": float(payload.get("Organic_Carbon_pct", 0.5)),
            "Avg_Temperature_C": float(payload.get("Avg_Temp_C", 25.0)), # From build_model_payload
            "Rainfall_mm": float(payload.get("Seasonal_Rainfall_mm", 0.0)), # From build_model_payload
            "Humidity_pct": float(payload.get("Avg_Humidity_pct", 50.0)), # From build_model_payload
        }

        # 2. One-Hot Encoding Construction
        # Initialize all OHE columns to 0
        for feature in self.model_features:
            if feature not in data: # It's a categorical dummy
                data[feature] = 0

        # Set District (with normalization)
        dist = payload.get("District", "")
        # Normalize district name if it's in the mapping
        dist = district_map.get(dist, dist)
        dist_col = f"District_{dist}"
        if dist_col in data:
            data[dist_col] = 1
        else:
            print(f"⚠️ Warning: District '{dist}' not found in model features. Available: {[f for f in data.keys() if f.startswith('District_')][:5]}...")
            
        # Set Season
        szn = payload.get("Season", "")
        szn_col = f"Season_{szn}"
        if szn_col in data:
            data[szn_col] = 1
            
        # Set Soil Type
        soil = payload.get("Soil_Type", "")
        soil_col = f"Soil_Type_{soil}"
        if soil_col in data:
            data[soil_col] = 1

        # Return as DataFrame with exact column order
        return pd.DataFrame([data])[self.model_features]

    def predict(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        
        X = self._build_feature_vector(payload)
        
        # Predict
        try:
            probabilities = self.model.predict_proba(X.values)[0]
        except AttributeError:
             # Fallback if model doesn't support probability
            pred = self.model.predict(X.values)[0]
            probabilities = np.zeros(len(self.classes))
            if pred in self.classes:
                probabilities[list(self.classes).index(pred)] = 1.0

        # Validate Length
        if len(probabilities) != len(self.classes):
            print(f"⚠️ Class mismatch! Model output {len(probabilities)} vs Classes {len(self.classes)}")
            # Fallback: Determine max index and hopefully it's within range
            # Or try to fix self.classes if it was loaded incorrectly
            pass

        # Map to classes
        top_indices = np.argsort(probabilities)[::-1][:3]
        results = []
        for idx in top_indices:
            # Safety check
            if idx >= len(self.classes):
                continue
                
            score = probabilities[idx]
            if score > 0.0:
                results.append({
                    "crop": self.classes[idx],
                    "score": float(score)
                })
        
        return results if results else [{"crop": "Rice", "score": 0.0}] # Fallback



@app.route("/")
def home() -> str:
    return render_template("landing.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        phone = request.form.get('phone')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(phone=phone).first()
        
        if user and user.check_password(password):
            if not user.is_verified:
                flash('Please verify your phone number first.', 'warning')
                # Regenerate OTP for them
                otp = generate_otp()
                otp_storage[phone] = otp
                print(f"🔐 SIMULATED OTP for {phone}: {otp}")
                return redirect(url_for('verify_otp', phone=phone))
                
            login_user(user, remember=remember)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid phone number or password.', 'error')
            
    return render_template('auth/login.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        phone = request.form.get('phone')
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        
        user = User.query.filter_by(phone=phone).first()
        if user:
            flash('Phone number already registered', 'error')
            return redirect(url_for('signup'))
            
        new_user = User(
            phone=phone,
            username=username,
            full_name=full_name
        )
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Generate OTP
        otp = generate_otp()
        otp_storage[phone] = otp
        print(f"🔐 SIMULATED OTP for {phone}: {otp}")
        
        flash('Account created! Please enter the OTP from the console.', 'success')
        return redirect(url_for('verify_otp', phone=phone))
        
    return render_template('auth/signup.html')

@app.route("/verify_otp", methods=['GET', 'POST'])
def verify_otp():
    phone = request.args.get('phone') or request.form.get('phone')
    
    if not phone:
        flash('No phone number provided.', 'error')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        stored_otp = otp_storage.get(phone)
        
        if stored_otp and entered_otp == stored_otp:
            user = User.query.filter_by(phone=phone).first()
            if user:
                user.is_verified = True
                db.session.commit()
                # Clean up OTP
                del otp_storage[phone]
                
                login_user(user)
                flash('Phone verified successfully! Welcome.', 'success')
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid OTP. Please check console.', 'error')
            
    return render_template('auth/verify_otp.html', phone=phone)



@app.route("/forgot_password", methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = user.generate_reset_token()
            db.session.commit()
            send_reset_email(email, token)
        flash('If an account exists with that email, a password reset link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.verify_reset_token(token):
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password == confirm_password:
            user.set_password(password)
            user.reset_token = None
            user.reset_token_expiry = None
            db.session.commit()
            flash('Your password has been reset successfully.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'error')
            
    return render_template('reset_password.html', token=token)

@app.route("/profile", methods=['GET', 'POST'])
# @login_required <-- Removed for Guest Access
def profile():
    # Handle Profile Updates
    if request.method == 'POST':
        if current_user.is_authenticated:
            # Update Authenticated User
            current_user.full_name = request.form.get('full_name')
            current_user.phone = request.form.get('phone')
            current_user.district = request.form.get('district')
            db.session.commit()
            flash('Profile updated successfully!', 'success')
        else:
            # Update Guest Profile in Session
            session.permanent = True # Ensure long-term persistence
            session["guest_profile"] = {
                "full_name": request.form.get("full_name"),
                "phone": request.form.get("phone"),
                "district": request.form.get("district")
            }
            flash("Profile updated successfully (Guest)", "success")
        return redirect(url_for('profile'))

    # Display Profile
    if current_user.is_authenticated:
        user_data = current_user
        prediction_count = Prediction.query.filter_by(user_id=current_user.id).count()
        days_member = (datetime.utcnow() - current_user.created_at).days
    else:
        # Create a mock user object from session for the template
        guest_data = session.get('guest_profile', {})
        class GuestUser:
            full_name = guest_data.get('full_name', 'Guest Farmer')
            phone = guest_data.get('phone', '')
            district = guest_data.get('district', '')
            profile_image = 'default_avatar.png'
            is_authenticated = False
            username = "Guest"
        user_data = GuestUser()
        prediction_count = 0
        days_member = 0

    return render_template('profile.html', user=user_data, prediction_count=prediction_count, days_member=days_member)

# Avatar update remains login protected for now, or could be hidden for guests

@app.route("/update_avatar", methods=['POST'])
@login_required
def update_avatar():
    if 'profile_image' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('profile'))
    
    file = request.files['profile_image']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('profile'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"user_{current_user.id}_{file.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        # Resize image
        resize_image(image_path)
        
        # Update user record
        current_user.profile_image = filename
        db.session.commit()
        
        flash('Profile picture updated successfully', 'success')
    else:
        flash('Invalid file type', 'error')
        
    return redirect(url_for('profile'))

@app.route("/select_avatar", methods=['POST'])
@login_required
def select_avatar():
    avatar = request.form.get('avatar')
    # Validate selection against allowed list
    allowed_avatars = [
        'avatar_male_rural.png', 
        'avatar_female_rural.png', 
        'avatar_male_modern.png', 
        'avatar_female_modern.png'
    ]
    if avatar in allowed_avatars:
        current_user.profile_image = avatar
        db.session.commit()
        flash('Avatar selected successfully!', 'success')
    else:
        flash('Invalid avatar selection.', 'error')
    return redirect(url_for('profile'))

@app.route("/update_profile", methods=['POST'])
@login_required
def update_profile():
    # Update personal details
    current_user.full_name = request.form.get('full_name')
    current_user.phone = request.form.get('phone')
    
    # Update farm details
    current_user.district = request.form.get('district')
    try:
        current_user.farm_size = float(request.form.get('farm_size')) if request.form.get('farm_size') else None
    except ValueError:
        pass
    current_user.primary_crops = request.form.get('primary_crops')
    
    # Password change
    new_password = request.form.get('new_password')
    if new_password:
        current_password = request.form.get('current_password')
        if current_user.check_password(current_password):
            if request.form.get('confirm_new_password') == new_password:
                current_user.set_password(new_password)
                flash('Password updated successfully', 'success')
            else:
                flash('New passwords do not match', 'error')
        else:
            flash('Incorrect current password', 'error')
            
    db.session.commit()
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

@app.route("/history")
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(desc(Prediction.created_at)).all()
    return render_template('history.html', predictions=predictions)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/dashboard", methods=['GET'])
# @login_required  <-- REMOVED per user request
def dashboard() -> str:
    return render_template("index.html", user=current_user)

@app.route("/about")
def about_page() -> str:
    return render_template("about.html")

@app.route("/weather")
def weather_page() -> str:
    return render_template("weather.html")

@app.route("/schemes-center")
def schemes_page() -> str:
    return render_template("schemes.html")



@app.route("/contact", methods=['GET', 'POST'])
def contact_page() -> str:
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        msg = ContactMessage(name=name, email=email, message=message)
        db.session.add(msg)
        db.session.commit()
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact_page'))
        
    return render_template("contact.html")

@app.route("/faq")
def faq_page() -> str:
    return render_template("faq.html")

@app.route("/privacy")
def privacy_page() -> str:
    return render_template("privacy.html")

@app.route("/terms")
def terms_page() -> str:
    return render_template("terms.html")

@app.route("/get_district_names")
def get_district_names() -> Any:
    return jsonify(district_service.get_districts())

@app.route("/get_district_data/<district_name>")
def get_district_data(district_name: str) -> Any:
    try:
        return jsonify(district_service.get_district_data(district_name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

@app.route("/auto_defaults")
def auto_defaults() -> Any:
    district = request.args.get("district")
    season = request.args.get("season")
    if not district:
        return jsonify({"error": "district is required"}), 400
    try:
        data = district_service.get_auto_defaults(district, season)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    response = chatbot_service.answer(user_message)
    return jsonify({"response": response})

@app.route("/api/nearest-district")
def get_nearest_district():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        
        # Simple Euclidean distance approximation (sufficient for district level)
        # or Haversine if accuracy needed. Euclidean is fine for local region.
        closest_district = None
        min_dist = float('inf')
        
        for district, coords in DISTRICT_COORDINATES.items():
            # coords is {"lat": ..., "lon": ...}
            d_lat = coords["lat"] - lat
            d_lon = coords["lon"] - lon
            dist_sq = d_lat*d_lat + d_lon*d_lon
            
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_district = district
                
        if closest_district:
            return jsonify({"district": closest_district})
        else:
            return jsonify({"error": "No district found"}), 404
            
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400

@app.route("/api/weather/<district>")
def weather(district: str) -> Any:
    data = weather_service.get_weather(district)
    if not data:
        return jsonify({"error": "Weather data unavailable"}), 404
    return jsonify(data)

@app.route("/schemes")
def schemes() -> Any:
    return jsonify(scheme_service.list_schemes())



@app.route("/suggestion")
@app.route("/suggestion/<crop_name>")
def suggestion_page(crop_name: str = None):
    """Unified suggestion page handling both URL params and query params."""
    district = request.args.get('district')
    crop = crop_name or request.args.get('crop')
    season = request.args.get('season', 'Kharif')
    
    if not district or not crop:
        # Fallback to history if parameters are missing
        if current_user.is_authenticated:
            last_p = Prediction.query.filter_by(user_id=current_user.id).order_by(desc(Prediction.created_at)).first()
            if last_p:
                district = district or last_p.district
                crop = crop or last_p.top_crop
                season = season or last_p.season
        
        # Absolute defaults if still missing
        district = district or "Guntur"
        crop = crop or "Paddy"
        season = season or "Kharif"

    # Fetch crop-specific guidance from dataset
    guidance = district_service.fetch_guidance(district, crop) or {}
    weather_data = weather_service.get_weather(district)
    schemes = [s for s in scheme_service.list_schemes()]
    
    # --- Data Gathering for AI & Graphs ---
    current_soil = {}
    last_prediction = None
    if current_user.is_authenticated:
        last_prediction = Prediction.query.filter_by(user_id=current_user.id).order_by(desc(Prediction.created_at)).first()

    if last_prediction and last_prediction.mode == 'manual':
        current_soil = {
            'N': last_prediction.soil_n or 0,
            'P': last_prediction.soil_p or 0,
            'K': last_prediction.soil_k or 0,
            'pH': last_prediction.soil_ph or 7.0,
            'soil_type': last_prediction.soil_type or 'Red',
            'water_source': last_prediction.water_source or 'Well'
        }
    
    if not current_soil.get('N'):
        defaults = district_service.get_auto_defaults(district, season)
        current_soil = {
            'N': defaults.get('soil_n', 0),
            'P': defaults.get('soil_p', 0),
            'K': defaults.get('soil_k', 0),
            'pH': defaults.get('soil_ph', 7.0),
            'soil_type': defaults.get('soil_type', 'Red'),
            'water_source': defaults.get('water_source', 'Well')
        }

    weather_summary = "No weather data available"
    if weather_data and "advice" in weather_data:
        weather_summary = "; ".join(weather_data["advice"])

    relevant_schemes_list = [s['title'] for s in schemes if crop.lower() in s['title'].lower() or crop.lower() in s['description'].lower()]
    if not relevant_schemes_list: relevant_schemes_list = [s['title'] for s in schemes[:4]]

    # --- CONSOLIDATED AI ADVICE ---
    ai_context = {
        "crop": crop, "district": district,
        "n": current_soil.get('N', 0), 
        "p": current_soil.get('P', 0), 
        "k": current_soil.get('K', 0), 
        "ph": current_soil.get('pH', 7.0),
        "soil_type": current_soil.get('soil_type', 'Red'), 
        "water_source": current_soil.get('water_source', 'Well'),
        "season": season, "weather": weather_summary,
        "relevant_schemes": relevant_schemes_list
    }
    ai_advice = ai_service.get_comprehensive_advice(ai_context)
    ai_fert = ai_advice.get("fertilizer", "")
    ai_irrig = ai_advice.get("irrigation", "")
    ai_market = ai_advice.get("market", "")
    ai_climate_7d = ai_advice.get("climate", "")
    ai_climate_1d = "" 

    # --- Generate Plots ---
    img_plan = guidance.get('fertilizer_plan') or {}
    optimal_soil = {
        'N': img_plan.get('N_kg_ha', 120),
        'P': img_plan.get('P_kg_ha', 60),
        'K': img_plan.get('K_kg_ha', 60),
        'pH': 6.5
    }

    fig_npk = go.Figure(data=[
        go.Bar(name='Your Soil', x=['Nitrogen', 'Phosphorus', 'Potassium'], 
               y=[current_soil['N'], current_soil['P'], current_soil['K']],
               marker_color='#94a3b8'),
        go.Bar(name='Recommended', x=['Nitrogen', 'Phosphorus', 'Potassium'], 
               y=[optimal_soil['N'], optimal_soil['P'], optimal_soil['K']],
               marker_color=['#16a34a', '#3b82f6', '#f59e0b'])
    ])
    fig_npk.update_layout(title="Soil NPK: Current vs Recommended (kg/ha)", barmode='group', height=350,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    npk_plot = fig_npk.to_html(full_html=False, include_plotlyjs='cdn')

    fig_ph = go.Figure(go.Indicator(
        mode = "number+gauge+delta", value = current_soil['pH'], delta = {'reference': optimal_soil['pH']},
        title = {'text': "Soil pH Status"},
        gauge = {'axis': {'range': [0, 14]}, 'bar': {'color': "#10b981" if 6 <= current_soil['pH'] <= 7.5 else "#f59e0b"},
                 'steps': [{'range': [0, 5.5], 'color': "#fee2e2"}, {'range': [5.5, 7.5], 'color': "#dcfce7"}, {'range': [7.5, 14], 'color': "#fee2e2"}],
                 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': optimal_soil['pH']}}
    ))
    fig_ph.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    ph_plot = fig_ph.to_html(full_html=False, include_plotlyjs=False)

    relevant_schemes = [s for s in schemes if crop.lower() in s['title'].lower() or crop.lower() in s['description'].lower()]
    if not relevant_schemes: relevant_schemes = schemes[:4]
    else: relevant_schemes = relevant_schemes[:4]

    return render_template(
        "suggestion.html",
        district=district, crop=crop, guidance=guidance, weather=weather_data,
        schemes=relevant_schemes, user=current_user,
        fertilizer_plan_gemini=ai_fert, irrigation_plan_gemini=ai_irrig,
        market_potential=ai_market, climate_1d=ai_climate_1d, climate_7d=ai_climate_7d,
        npk_plot=npk_plot, ph_plot=ph_plot, season=season
    )

@app.route("/predict", methods=["GET", "POST"])
# @login_required  <-- REMOVED per user request
def predict() -> Any:
    # If GET request, redirect to dashboard with message
    if request.method == "GET":
        mode = request.args.get("mode", "manual")
        flash(f'Please use the {mode.title()} Mode form below to get predictions.', 'info')
        return redirect(url_for('dashboard'))
    
    # POST request - process prediction
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Invalid input payload"}), 400

    district = payload.get("district")
    if not district:
        return jsonify({"error": "Please select a district"}), 400

    season = payload.get("season")
    mode = payload.get("mode", "manual").lower()
    if mode not in {"manual", "auto"}:
        return jsonify({"error": "Mode must be 'manual' or 'auto'"}), 400

    try:
        model_payload = district_service.build_model_payload(
            district=district,
            season=season,
            raw_payload=payload,
            mode=mode,
        )
        recommendations = recommendation_engine.predict(model_payload)
        top_crop = recommendations[0]["crop"]
        guidance = district_service.fetch_guidance(district, top_crop)
        location_snapshot = district_service.get_auto_defaults(district, season)
        weather_snapshot = weather_service.get_weather(district)

        # Save prediction to database ONLY if user is logged in
        if current_user.is_authenticated:
            try:
                prediction = Prediction(
                    user_id=current_user.id,
                    district=district,
                    mandal=model_payload.get("Mandal"),
                    season=model_payload.get("Season"),
                    soil_type=model_payload.get("Soil_Type"),
                    water_source=model_payload.get("Water_Source"),
                    mode=mode,
                    top_crop=recommendations[0]["crop"],
                    top_crop_score=recommendations[0]["score"],
                    second_crop=recommendations[1]["crop"] if len(recommendations) > 1 else None,
                    second_crop_score=recommendations[1]["score"] if len(recommendations) > 1 else None,
                    third_crop=recommendations[2]["crop"] if len(recommendations) > 2 else None,
                    third_crop_score=recommendations[2]["score"] if len(recommendations) > 2 else None,
                )
                db.session.add(prediction)
                db.session.commit()
            except Exception as e:
                print(f"Error saving prediction: {e}")

        # OpenAI Enhancements for the top result
        top_crop_name = recommendations[0]["crop"]
        market_pot = ai_service.get_market_potential(top_crop_name, district)
        
        response_payload = {
            "mode": mode,
            "recommendations": recommendations,
            "location_details": {
                "district": district,
                "mandal": model_payload.get("Mandal"),
                "season": model_payload.get("Season"),
                "soil_type": model_payload.get("Soil_Type"),
                "rainfall": location_snapshot.get("rainfall"),
                "humidity": location_snapshot.get("humidity"),
            },
            "guidance": guidance,
            "weather": weather_snapshot,
            "auto_defaults": location_snapshot,
            "market_potential": market_pot, # Add this field
        }
        return jsonify(response_payload)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("❌ PREDICTION ERROR:", str(e))
        print(err_msg)
        try:
            with open("error.log", "w", encoding="utf-8") as f:
                f.write(err_msg)
        except:
            pass
        return jsonify({"error": "Unable to generate recommendations at the moment."}), 500

# REDUNDANT ROUTE REMOVED (Consolidated above)
    
    return render_template(
        "suggestion.html",
        crop_name=crop_name,
        district=district,
        season=season,
        guidance=guidance,
        weather=weather_data,
        schemes=relevant_schemes,
        npk_plot=npk_plot,
        ph_plot=ph_plot,
        fertilizer_plan_gemini=ai_fert,
        irrigation_plan_gemini=ai_irrig,
        market_potential=ai_market,
        climate_1d=ai_climate_1d,
        climate_7d=ai_climate_7d
    )


@app.route("/api/suggest", methods=["POST"])
def api_suggest() -> Any:
    """
    Get suggestions for a specific district, season, and crop.
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Invalid input payload"}), 400

    district = payload.get("district")
    season = payload.get("season")
    crop = payload.get("crop")

    if not all([district, season, crop]):
        return jsonify({"error": "district, season, and crop are required"}), 400

    # 1. Fetch Guidance (Fertilizer, Irrigation)
    guidance = district_service.fetch_guidance(district, crop)

    # 2. Fetch Weather
    weather_snapshot = weather_service.get_weather(district)
    
    # 3. Fetch Schemes (Filtered by crop/district if possible, for now return all or top 4)
    # The user asked for "Schemes card: filter schemes using both crop and district"
    # Current SchemeService just lists all. We'll refine this later or client-side filter.
    schemes = scheme_service.list_schemes()[:4] # Return top 4 for now

    # --- AI INTEGRATIONS ---
    # Extract actual soil data from defaults or params if possible
    defaults = district_service.get_auto_defaults(district, season)
    n = defaults.get('soil_n', 0)
    p = defaults.get('soil_p', 0)
    k = defaults.get('soil_k', 0)
    ph = defaults.get('soil_ph', 7.0)
    soil_type = defaults.get('soil_type', 'Red')
    water_source = defaults.get('water_source', 'Well')

    weather_data_raw = weather_service.get_weather(district)
    weather_summary = "No weather data available"
    if weather_data_raw and "advice" in weather_data_raw:
        weather_summary = "; ".join(weather_data_raw["advice"])

    # --- CONSOLIDATED AI ADVICE ---
    ai_context = {
        "crop": crop, "district": district,
        "n": n, "p": p, "k": k, "ph": ph,
        "soil_type": soil_type, "water_source": water_source,
        "season": season, "weather": weather_summary
    }
    ai_advice = ai_service.get_comprehensive_advice(ai_context)
    ai_fert = ai_advice.get("fertilizer", "")
    ai_irrig = ai_advice.get("irrigation", "")
    ai_market = ai_advice.get("market", "")
    ai_climate_7d = ai_advice.get("climate", "")
    ai_climate_1d = "" 
    
    # Overwrite/Augment guidance with AI data
    guidance["fertilizer_plan_gemini"] = ai_fert
    guidance["irrigation_plan_gemini"] = ai_irrig

    # 4. Mock Graphs Data or fetch if available
    graphs_data = {
        "soil_suitability": {
            "N": {"current": 0, "optimal": 0}, # Placeholder
            "P": {"current": 0, "optimal": 0},
            "K": {"current": 0, "optimal": 0},
            "pH": {"current": 0, "optimal": 0}
        }
    }

    # 5. Crop Image (Use static path convention)
    # Convention: /static/crops/{crop_name}.jpg or similar.
    # We can provide a helper URL.
    
    response = {
        "crop": crop,
        "district": district,
        "season": season,
        "fertilizer_plan": guidance.get("fertilizer_plan", {}),
        "irrigation_plan": guidance.get("irrigation_plan", {}),
        "fertilizer_plan_gemini": ai_fert,
        "irrigation_plan_gemini": ai_irrig,
        "market_potential": ai_market,
        "climate_1d": ai_climate_1d,
        "climate_7d": ai_climate_7d,
        "weather_summary": weather_snapshot,
        "schemes": schemes,
        "crop_image": url_for('static', filename=f'crops/{crop.lower()}.jpg'),
        "graphs_data": graphs_data
    }
    return jsonify(response)



@app.route("/login/google")
def login_google():
    flash("Continuing with Google requires valid API credentials. For this preview, please use the standard login or signup. OAuth framework is fully ready in models.py and requirements.txt.", "info")
    return redirect(url_for("login"))

@app.route("/login/facebook")
def login_facebook():
    flash("Continuing with Facebook requires valid API credentials. For this preview, please use the standard login or signup.", "info")
    return redirect(url_for("login"))

# Initialize Services (MUST BE AT BOTTOM, after classes are defined)
DATASET = pd.read_csv(DATASET_PATH)

# --- PATCH: Normalize Dataset Columns ---
# Rename columns to match what app.py expects
DATASET = DATASET.rename(columns={
    "Avg_Temperature_C": "Avg_Temp_C",
    "Rainfall_mm": "Seasonal_Rainfall_mm",
    "Humidity_pct": "Avg_Humidity_pct"
})

# Add missing columns with defaults if they don't exist
if "Mandal" not in DATASET.columns:
    DATASET["Mandal"] = "Unknown"
if "Water_Source" not in DATASET.columns:
    DATASET["Water_Source"] = "Well"
if "Secondary_Crop" not in DATASET.columns:
    DATASET["Secondary_Crop"] = "None"
# ----------------------------------------
district_service = DistrictDataService(DATASET)
recommendation_engine = CropRecommendationEngine(DATASET)
scheme_service = SchemeService(GOVERNMENT_SCHEMES)
chatbot_service = ChatbotService(CHATBOT_KNOWLEDGE)
weather_service = WeatherService(DISTRICT_COORDINATES)
ai_service = AIService()

if __name__ == "__main__":

    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
