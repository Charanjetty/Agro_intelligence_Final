# User Authentication and Database Models
# This file will be imported by app.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import secrets

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(15), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    full_name = db.Column(db.String(100))
    district = db.Column(db.String(50))
    
    # Phone verification
    is_verified = db.Column(db.Boolean, default=False)
    
    # Password reset
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiry = db.Column(db.DateTime)
    
    # OAuth integration
    oauth_provider = db.Column(db.String(20))  # 'google', 'facebook', None
    oauth_id = db.Column(db.String(200))
    
    # Profile information
    profile_image = db.Column(db.String(200), default='default_avatar.png')
    farm_size = db.Column(db.Float)  # in acres
    primary_crops = db.Column(db.String(200))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches"""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    def generate_reset_token(self):
        """Generate password reset token"""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiry = datetime.utcnow() + timedelta(hours=24)
        return self.reset_token
    
    def verify_reset_token(self, token):
        """Verify if reset token is valid and not expired"""
        if not self.reset_token or self.reset_token != token:
            return False
        if datetime.utcnow() > self.reset_token_expiry:
            return False
        return True
    
    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Model to store user predictions"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Input parameters
    district = db.Column(db.String(50), nullable=False)
    mandal = db.Column(db.String(50))
    season = db.Column(db.String(20))
    soil_type = db.Column(db.String(30))
    water_source = db.Column(db.String(30))
    mode = db.Column(db.String(10))  # manual or auto
    
    # Soil parameters (for manual mode)
    soil_ph = db.Column(db.Float)
    organic_carbon = db.Column(db.Float)
    soil_n = db.Column(db.Float)
    soil_p = db.Column(db.Float)
    soil_k = db.Column(db.Float)
    
    # Results
    top_crop = db.Column(db.String(50))
    top_crop_score = db.Column(db.Float)
    second_crop = db.Column(db.String(50))
    second_crop_score = db.Column(db.Float)
    third_crop = db.Column(db.String(50))
    third_crop_score = db.Column(db.Float)
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.top_crop}>'


class ContactMessage(db.Model):
    """Model to store contact form submissions"""
    __tablename__ = 'contact_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(15))
    subject = db.Column(db.String(200))
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='new')  # new, read, replied
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContactMessage {self.id} from {self.name}>'
