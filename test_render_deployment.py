# Test script for Render deployment
import requests
import json

def test_render_deployment(base_url):
    """Test all endpoints of the deployed DeepCSAT API"""
    
    print(f"🧪 Testing DeepCSAT API at: {base_url}")
    print("="*60)
    
    # Test 1: Home endpoint
    print("1. Testing Home endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Home endpoint working")
            print(f"   Response: {response.json()['message']}")
        else:
            print(f"❌ Home endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Home endpoint error: {e}")
    
    print()
    
    # Test 2: Health check
    print("2. Testing Health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Models loaded: {health_data['models_loaded']}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    print()
    
    # Test 3: Models endpoint
    print("3. Testing Models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            print("✅ Models endpoint working")
            models_data = response.json()
            print(f"   Available models: {models_data['available_models']}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
    
    print()
    
    # Test 4: Prediction endpoint
    print("4. Testing Prediction endpoint...")
    try:
        test_data = {
            "channel_name": "Inbound",
            "category": "Product Queries",
            "item_price": 150.0,
            "handling_time": 5.0,
            "agent_experience": ">90"
        }
        
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            print("✅ Prediction endpoint working")
            prediction_data = response.json()
            print(f"   Predicted CSAT Score: {prediction_data['prediction']}")
            print(f"   Model used: {prediction_data['model_used']}")
            print(f"   Confidence: {prediction_data['confidence']}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
    
    print()
    
    # Test 5: Documentation endpoint
    print("5. Testing Documentation endpoint...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ Documentation endpoint working")
            docs_data = response.json()
            print(f"   API Title: {docs_data['title']}")
            print(f"   Version: {docs_data['version']}")
        else:
            print(f"❌ Documentation endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Documentation endpoint error: {e}")
    
    print()
    print("="*60)
    print("🎉 Testing completed!")

def test_batch_predictions(base_url):
    """Test batch prediction endpoint"""
    print("\n6. Testing Batch Predictions...")
    try:
        batch_data = {
            "records": [
                {
                    "id": "1",
                    "channel_name": "Inbound",
                    "category": "Product Queries"
                },
                {
                    "id": "2", 
                    "channel_name": "Email",
                    "category": "Returns"
                }
            ]
        }
        
        response = requests.post(f"{base_url}/predict/batch", json=batch_data)
        if response.status_code == 200:
            print("✅ Batch prediction endpoint working")
            batch_result = response.json()
            print(f"   Predictions count: {batch_result['count']}")
            for i, pred in enumerate(batch_result['predictions']):
                print(f"   Record {i+1}: CSAT Score = {pred['prediction']}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")

if __name__ == "__main__":
    # Replace with your actual Render URL
    RENDER_URL = "https://your-app-name.onrender.com"
    
    print("DeepCSAT Render Deployment Test")
    print("="*60)
    print(f"Testing URL: {RENDER_URL}")
    print("Make sure to replace 'your-app-name' with your actual Render app name")
    print()
    
    # Test all endpoints
    test_render_deployment(RENDER_URL)
    
    # Test batch predictions
    test_batch_predictions(RENDER_URL)
    
    print("\n📝 Instructions:")
    print("1. Replace 'your-app-name' in the RENDER_URL with your actual app name")
    print("2. Make sure your app is deployed and running on Render")
    print("3. Run this script: python test_render_deployment.py")
    print("4. Check all endpoints are working correctly")
