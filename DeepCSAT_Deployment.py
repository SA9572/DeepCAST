# DeepCSAT: Production Deployment Script
# Complete deployment pipeline for CSAT prediction system

import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DeepCSATDeployment:
    """
    Production deployment system for DeepCSAT project
    Handles model serving, prediction API, and monitoring
    """
    
    def __init__(self, model_path='models/', config_path='config/'):
        self.model_path = model_path
        self.config_path = config_path
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metadata = {}
        
        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        try:
            # Load traditional ML models
            model_files = {
                'Random Forest': 'random_forest.joblib',
                'Gradient Boosting': 'gradient_boosting.joblib',
                'SVR': 'svr.joblib',
                'Linear Regression': 'linear_regression.joblib',
                'Ridge Regression': 'ridge_regression.joblib',
                'Lasso Regression': 'lasso_regression.joblib'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"✓ Loaded {model_name}")
                else:
                    print(f"✗ {model_name} not found at {filepath}")
            
            # Load deep learning model
            dl_model_path = os.path.join(self.model_path, 'deep_learning_ann.h5')
            if os.path.exists(dl_model_path):
                import tensorflow as tf
                self.models['Deep Learning ANN'] = tf.keras.models.load_model(dl_model_path)
                print("✓ Loaded Deep Learning ANN")
            else:
                print("✗ Deep Learning ANN not found")
            
            # Load scaler and encoders
            scaler_path = os.path.join(self.model_path, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Loaded scaler")
            
            encoders_path = os.path.join(self.model_path, 'label_encoders.joblib')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                print("✓ Loaded label encoders")
            
            print("Model loading completed!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_model_metadata(self, metadata):
        """Save model metadata"""
        self.model_metadata = metadata
        
        metadata_file = os.path.join(self.config_path, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model metadata saved to {metadata_file}")
    
    def load_model_metadata(self):
        """Load model metadata"""
        metadata_file = os.path.join(self.config_path, 'model_metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            print("Model metadata loaded successfully")
        else:
            print("Model metadata not found")
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            # Convert to DataFrame if not already
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            else:
                input_df = input_data.copy()
            
            # Apply label encoding for categorical variables
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
            
            # Select only required features
            if self.feature_columns:
                input_df = input_df[self.feature_columns]
            
            # Handle missing values
            input_df = input_df.fillna(input_df.median())
            
            return input_df
            
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            return None
    
    def predict(self, input_data, model_name='Deep Learning ANN', return_confidence=False):
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            if processed_data is None:
                return None
            
            model = self.models[model_name]
            
            # Make prediction based on model type
            if model_name == 'Deep Learning ANN':
                # Scale data for deep learning model
                scaled_data = self.scaler.transform(processed_data)
                prediction = model.predict(scaled_data).flatten()
            elif model_name in ['SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                # Scale data for these models
                scaled_data = self.scaler.transform(processed_data)
                prediction = model.predict(scaled_data)
            else:
                # No scaling needed for tree-based models
                prediction = model.predict(processed_data)
            
            # Round prediction to nearest integer (CSAT scores are 1-5)
            prediction = np.round(np.clip(prediction, 1, 5)).astype(int)
            
            result = {
                'prediction': prediction.tolist(),
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_confidence and hasattr(model, 'predict_proba'):
                # Get prediction confidence if available
                confidence = model.predict_proba(processed_data)
                result['confidence'] = confidence.tolist()
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def batch_predict(self, input_data, model_name='Deep Learning ANN'):
        """Make batch predictions"""
        try:
            results = []
            
            for i, row in input_data.iterrows():
                prediction = self.predict(row.to_dict(), model_name)
                if prediction:
                    results.append(prediction)
            
            return results
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None
    
    def get_model_performance(self):
        """Get model performance metrics"""
        if not self.model_metadata:
            self.load_model_metadata()
        
        return self.model_metadata.get('performance_metrics', {})
    
    def create_prediction_api(self, host='0.0.0.0', port=5000):
        """Create Flask API for predictions"""
        try:
            from flask import Flask, request, jsonify
            import traceback
            
            app = Flask(__name__)
            
            @app.route('/health', methods=['GET'])
            def health_check():
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'models_loaded': list(self.models.keys())
                })
            
            @app.route('/predict', methods=['POST'])
            def predict_endpoint():
                try:
                    data = request.get_json()
                    
                    if not data:
                        return jsonify({'error': 'No data provided'}), 400
                    
                    # Get model name from request or use default
                    model_name = data.get('model_name', 'Deep Learning ANN')
                    
                    # Make prediction
                    result = self.predict(data, model_name)
                    
                    if result is None:
                        return jsonify({'error': 'Prediction failed'}), 500
                    
                    return jsonify(result)
                    
                except Exception as e:
                    return jsonify({
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }), 500
            
            @app.route('/models', methods=['GET'])
            def list_models():
                return jsonify({
                    'available_models': list(self.models.keys()),
                    'default_model': 'Deep Learning ANN'
                })
            
            @app.route('/model_info/<model_name>', methods=['GET'])
            def model_info(model_name):
                if model_name not in self.models:
                    return jsonify({'error': 'Model not found'}), 404
                
                model = self.models[model_name]
                info = {
                    'model_name': model_name,
                    'model_type': type(model).__name__,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add model-specific information
                if hasattr(model, 'n_features_in_'):
                    info['n_features'] = model.n_features_in_
                if hasattr(model, 'feature_importances_'):
                    info['has_feature_importance'] = True
                
                return jsonify(info)
            
            print(f"Starting Flask API server on {host}:{port}")
            app.run(host=host, port=port, debug=False)
            
        except ImportError:
            print("Flask not available. Install with: pip install flask")
        except Exception as e:
            print(f"Error creating API: {e}")
    
    def create_dockerfile(self):
        """Create Dockerfile for containerization"""
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories
RUN mkdir -p models config logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=DeepCSAT_Deployment.py

# Run the application
CMD ["python", "DeepCSAT_Deployment.py"]
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        print("Dockerfile created successfully")
    
    def create_docker_compose(self):
        """Create docker-compose.yml for easy deployment"""
        compose_content = """
version: '3.8'

services:
  deepcsat-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    
  deepcsat-monitor:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
"""
        
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        print("docker-compose.yml created successfully")
    
    def create_monitoring_config(self):
        """Create monitoring configuration"""
        os.makedirs('monitoring', exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'deepcsat-api'
    static_configs:
      - targets: ['deepcsat-api:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        with open('monitoring/prometheus.yml', 'w') as f:
            f.write(prometheus_config)
        
        print("Monitoring configuration created")
    
    def create_requirements_file(self):
        """Create production requirements file"""
        requirements = """
# DeepCSAT Production Requirements
flask==2.3.3
gunicorn==21.2.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
joblib==1.3.2
prometheus-client==0.17.1
"""
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        
        print("Production requirements file created")
    
    def create_deployment_script(self):
        """Create deployment script"""
        deploy_script = """#!/bin/bash

# DeepCSAT Deployment Script

echo "Starting DeepCSAT deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p models config logs monitoring

# Build and start services
echo "Building and starting services..."
docker-compose up -d --build

# Wait for services to start
echo "Waiting for services to start..."
sleep 30

# Check if API is healthy
echo "Checking API health..."
curl -f http://localhost:5000/health || echo "API health check failed"

echo "Deployment completed!"
echo "API available at: http://localhost:5000"
echo "Prometheus available at: http://localhost:9090"
"""
        
        with open('deploy.sh', 'w') as f:
            f.write(deploy_script)
        
        # Make script executable
        os.chmod('deploy.sh', 0o755)
        
        print("Deployment script created")
    
    def create_test_script(self):
        """Create test script for API"""
        test_script = """
import requests
import json

# Test API endpoints
base_url = "http://localhost:5000"

def test_health():
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_models():
    response = requests.get(f"{base_url}/models")
    print(f"Models list: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_prediction():
    # Sample input data
    sample_data = {
        "channel_name": "Inbound",
        "category": "Product Queries",
        "Sub-category": "Product Specific Information",
        "Customer Remarks": "Great service, very helpful",
        "Item_price": 150.0,
        "connected_handling_time": 5.0,
        "Agent_name": "John Doe",
        "Tenure Bucket": ">90",
        "Agent Shift": "Morning"
    }
    
    response = requests.post(f"{base_url}/predict", json=sample_data)
    print(f"Prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing DeepCSAT API...")
    test_health()
    test_models()
    test_prediction()
"""
        
        with open('test_api.py', 'w') as f:
            f.write(test_script)
        
        print("Test script created")
    
    def setup_production_environment(self):
        """Setup complete production environment"""
        print("Setting up production environment...")
        
        # Create all necessary files
        self.create_dockerfile()
        self.create_docker_compose()
        self.create_monitoring_config()
        self.create_requirements_file()
        self.create_deployment_script()
        self.create_test_script()
        
        print("Production environment setup completed!")
        print("\nTo deploy:")
        print("1. Ensure models are in the 'models/' directory")
        print("2. Run: ./deploy.sh")
        print("3. Test: python test_api.py")
    
    def generate_documentation(self):
        """Generate API documentation"""
        doc_content = """
# DeepCSAT API Documentation

## Overview
DeepCSAT API provides customer satisfaction score prediction for e-commerce platforms.

## Endpoints

### Health Check
- **GET** `/health`
- Returns API health status and loaded models

### List Models
- **GET** `/models`
- Returns list of available models

### Model Information
- **GET** `/model_info/<model_name>`
- Returns detailed information about a specific model

### Prediction
- **POST** `/predict`
- Makes CSAT score prediction

#### Request Body
```json
{
    "channel_name": "Inbound",
    "category": "Product Queries",
    "Sub-category": "Product Specific Information",
    "Customer Remarks": "Great service, very helpful",
    "Item_price": 150.0,
    "connected_handling_time": 5.0,
    "Agent_name": "John Doe",
    "Tenure Bucket": ">90",
    "Agent Shift": "Morning",
    "model_name": "Deep Learning ANN"
}
```

#### Response
```json
{
    "prediction": [5],
    "model_used": "Deep Learning ANN",
    "timestamp": "2023-12-01T10:30:00"
}
```

## Deployment

### Using Docker
```bash
./deploy.sh
```

### Manual Deployment
```bash
pip install -r requirements.txt
python DeepCSAT_Deployment.py
```

## Monitoring
- Prometheus metrics available at `/metrics`
- Prometheus UI at `http://localhost:9090`

## Testing
```bash
python test_api.py
```
"""
        
        with open('API_DOCUMENTATION.md', 'w') as f:
            f.write(doc_content)
        
        print("API documentation generated")

# Example usage
if __name__ == "__main__":
    print("DeepCSAT Deployment System Ready!")
    print("Usage:")
    print("1. Initialize: deployment = DeepCSATDeployment()")
    print("2. Load models: deployment.load_models()")
    print("3. Setup production: deployment.setup_production_environment()")
    print("4. Start API: deployment.create_prediction_api()")
