# 🧠 Alzheimer's Detection App - Next.js

A modern, AI-powered web application for Alzheimer's risk assessment using ensemble machine learning models. Built with Next.js 14, TypeScript, and Tailwind CSS.

## ✨ Features

- **Dual Model Predictions**: Voting and Stacking ensemble models
- **Modern UI**: Beautiful, responsive design with Tailwind CSS
- **Real-time Analysis**: Instant predictions with confidence scores
- **Comprehensive Input**: 30+ health and lifestyle factors
- **Model Comparison**: Side-by-side results from both models
- **Professional Design**: Clean, medical-grade interface

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- Python 3.8+ with scikit-learn
- Your trained ML models (`alzheimers_voting_model.pkl` and `alzheimers_stacking_model.pkl`)

### Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Install Python dependencies:**
   ```bash
   pip install scikit-learn numpy joblib
   ```

3. **Ensure your ML models are in the project root:**
   - `alzheimers_voting_model.pkl`
   - `alzheimers_stacking_model.pkl`

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🏗️ Architecture

### Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS with custom components
- **Icons**: Lucide React
- **TypeScript**: Full type safety

### Backend (API Routes)
- **API**: Next.js API routes (`/api/predict`)
- **ML Integration**: Python shell integration
- **Data Processing**: Real-time feature preprocessing

### Machine Learning
- **Models**: Voting and Stacking ensemble classifiers
- **Features**: 39 engineered features
- **Preprocessing**: Automated feature engineering
- **Output**: Risk probabilities and confidence scores

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Voting Ensemble | 91.86% | 88.74% | 88.16% | 88.45% | 93.47% |
| Stacking Ensemble | 93.49% | 90.26% | 91.45% | 90.85% | 94.44% |

## 🎯 Key Features

### Input Categories
- **Basic Information**: Age, gender, ethnicity, education, BMI
- **Lifestyle**: Smoking, alcohol, physical activity, diet, sleep
- **Health History**: Family history, cardiovascular, diabetes, depression
- **Vital Signs**: Blood pressure, cholesterol levels
- **Cognitive Assessment**: MMSE score, functional assessment, symptoms

### Output Features
- **Risk Assessment**: High/Low risk predictions
- **Confidence Scores**: Model confidence levels
- **Model Agreement**: Consensus between models
- **Probability Scores**: Detailed risk percentages

## 🛠️ Development

### Project Structure
```
├── app/
│   ├── api/predict/route.ts    # ML prediction API
│   ├── globals.css             # Global styles
│   ├── layout.tsx              # Root layout
│   └── page.tsx                # Main application
├── ml_predictor.py             # Python ML handler
├── package.json                # Dependencies
├── tailwind.config.js          # Tailwind configuration
└── tsconfig.json               # TypeScript config
```

### API Endpoint
**POST** `/api/predict`

**Request Body:**
```json
{
  "age": 65,
  "gender": "Male",
  "ethnicity": "Caucasian",
  "education": "Bachelor's",
  "bmi": 25.0,
  // ... other fields
}
```

**Response:**
```json
{
  "voting_prediction": 0,
  "stacking_prediction": 0,
  "voting_probabilities": {
    "low_risk": 0.79,
    "high_risk": 0.21
  },
  "stacking_probabilities": {
    "low_risk": 0.87,
    "high_risk": 0.13
  },
  "models_agree": true,
  "feature_count": 39
}
```

## 🎨 Customization

### Styling
- Modify `app/globals.css` for global styles
- Update `tailwind.config.js` for theme customization
- Component styles in `app/page.tsx`

### Model Integration
- Update `ml_predictor.py` for different models
- Modify feature engineering in the `preprocess_input` method
- Adjust API route in `app/api/predict/route.ts`

## 🚀 Deployment

### Vercel (Recommended)
1. Push to GitHub
2. Connect to Vercel
3. Deploy automatically

**Note**: The current implementation uses mock data for predictions when deployed to Vercel, as Python dependencies are not available in the Vercel environment. For a production deployment with real ML predictions, you would need to:

1. Deploy the Python ML models as a separate API service
2. Update the `/api/predict` route to call that external service
3. Configure environment variables for the external API endpoint

### Other Platforms
- **Netlify**: Static export with API functions
- **Railway**: Full-stack deployment
- **Docker**: Containerized deployment

## 📝 License

This project is for educational and research purposes. Please ensure compliance with medical regulations when using for clinical purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
