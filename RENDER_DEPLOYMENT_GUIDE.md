# DeepCSAT Render Deployment Guide

## 🚀 Deploy DeepCSAT to Render

This guide will help you deploy your DeepCSAT project to Render, a cloud platform that makes it easy to deploy web applications.

## 📋 Prerequisites

1. **GitHub Repository**: Your DeepCSAT project should be on GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **GitHub Account**: Connected to Render

## 🛠️ Step-by-Step Deployment

### Step 1: Prepare Your Repository

Your repository should have these files:
- ✅ `app.py` - Main Flask application
- ✅ `requirements_render.txt` - Python dependencies
- ✅ `Procfile` - Process file for Render
- ✅ `runtime.txt` - Python version specification
- ✅ `render.yaml` - Render configuration (optional)

### Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with your GitHub account
4. Authorize Render to access your repositories

### Step 3: Deploy from GitHub

1. **Log into Render Dashboard**
   - Go to your Render dashboard
   - Click "New +" button
   - Select "Web Service"

2. **Connect Repository**
   - Choose "Build and deploy from a Git repository"
   - Select your DeepCSAT repository
   - Click "Connect"

3. **Configure Deployment**
   - **Name**: `deepcsat-api` (or your preferred name)
   - **Environment**: `Python 3`
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (uses root)
   - **Build Command**: `pip install -r requirements_render.txt`
   - **Start Command**: `python app.py`

4. **Advanced Settings** (Optional)
   - **Python Version**: `3.11.0`
   - **Instance Type**: `Free` (for testing)
   - **Auto-Deploy**: `Yes` (deploys on every push)

5. **Environment Variables** (Optional)
   - `FLASK_ENV`: `production`
   - `PYTHON_VERSION`: `3.11.0`

6. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)

### Step 4: Test Your Deployment

Once deployed, you'll get a URL like: `https://deepcsat-api.onrender.com`

Test these endpoints:
- **Home**: `https://your-app.onrender.com/`
- **Health**: `https://your-app.onrender.com/health`
- **Models**: `https://your-app.onrender.com/models`
- **Docs**: `https://your-app.onrender.com/docs`

### Step 5: Test Predictions

Send a POST request to `/predict`:

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "channel_name": "Inbound",
    "category": "Product Queries",
    "item_price": 150.0,
    "handling_time": 5.0,
    "agent_experience": ">90"
  }'
```

Expected response:
```json
{
  "prediction": 4,
  "model_used": "Random Forest",
  "confidence": 0.85,
  "timestamp": "2023-12-01T10:30:00",
  "input_features": {
    "channel_name": "Inbound",
    "category": "Product Queries",
    "item_price": 150.0,
    "handling_time": 5.0,
    "agent_experience": ">90"
  }
}
```

## 🔧 Configuration Files Explained

### `app.py`
- Main Flask application
- Simplified for production deployment
- Includes all necessary endpoints
- Handles errors gracefully

### `requirements_render.txt`
- Minimal dependencies for Render
- Optimized for production
- Fast installation

### `Procfile`
- Tells Render how to start your app
- Uses `python app.py`

### `runtime.txt`
- Specifies Python version
- Ensures compatibility

### `render.yaml` (Optional)
- Advanced configuration
- Can be used instead of manual setup

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | API documentation |

## 📊 Example Usage

### Python Client
```python
import requests

# API endpoint
url = "https://your-app.onrender.com/predict"

# Sample data
data = {
    "channel_name": "Inbound",
    "category": "Product Queries",
    "item_price": 150.0,
    "handling_time": 5.0,
    "agent_experience": ">90"
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"Predicted CSAT Score: {result['prediction']}")
```

### JavaScript Client
```javascript
const url = "https://your-app.onrender.com/predict";
const data = {
    channel_name: "Inbound",
    category: "Product Queries",
    item_price: 150.0,
    handling_time: 5.0,
    agent_experience: ">90"
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('Predicted CSAT Score:', result.prediction);
});
```

## 🔍 Monitoring and Logs

1. **View Logs**
   - Go to your service dashboard
   - Click "Logs" tab
   - Monitor real-time logs

2. **Health Monitoring**
   - Render automatically monitors your app
   - Sends alerts if service goes down
   - Provides uptime statistics

3. **Performance Metrics**
   - Response times
   - Error rates
   - Resource usage

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check `requirements_render.txt`
   - Ensure all dependencies are compatible
   - Check Python version in `runtime.txt`

2. **App Crashes**
   - Check logs for error messages
   - Verify all imports work
   - Test locally first

3. **Slow Response**
   - Free tier has cold starts
   - Consider upgrading to paid plan
   - Optimize your code

4. **Memory Issues**
   - Free tier has limited memory
   - Optimize model loading
   - Use lighter models

### Debug Commands

```bash
# Test locally
python app.py

# Check dependencies
pip install -r requirements_render.txt

# Test API locally
curl http://localhost:5000/health
```

## 💰 Pricing

- **Free Tier**: 
  - 750 hours/month
  - Sleeps after 15 minutes of inactivity
  - Perfect for demos and testing

- **Paid Plans**:
  - Always-on service
  - More resources
  - Better performance

## 🔄 Updates and Maintenance

1. **Automatic Deployments**
   - Push to GitHub triggers deployment
   - No manual intervention needed

2. **Manual Deployments**
   - Go to service dashboard
   - Click "Manual Deploy"
   - Select branch/commit

3. **Rollbacks**
   - Go to "Deploys" tab
   - Click on previous deployment
   - Click "Redeploy"

## 🎉 Success!

Once deployed, your DeepCSAT API will be:
- ✅ Accessible worldwide
- ✅ Auto-scaling
- ✅ Monitored
- ✅ Easy to update
- ✅ Production-ready

## 📞 Support

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **Render Support**: [render.com/support](https://render.com/support)
- **GitHub Issues**: Create issues in your repository

---

**Congratulations! Your DeepCSAT project is now live on the web! 🚀**
