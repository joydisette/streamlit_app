# CN Forecasting Dashboard

A Streamlit-based interactive dashboard for visualizing and analyzing CN's forecasting data. This dashboard combines historical data visualization, forecasting insights, and an AI-powered assistant to help users understand the predictions and their underlying factors.

## 🚂 Features

- **Interactive Time Series Visualization**
  - Historical data display
  - Forecast predictions
  - Hover-based detailed information
  - Attribution scores for key factors

- **Quarterly Forecasts Display**
  - Clear visualization of Q1, Q2, and Q3 forecasts
  - Generation date tracking
  - CN-branded styling

- **AI Assistant Integration**
  - Pre-configured responses for common questions
  - Gemini AI integration (configurable)
  - Interactive chat interface
  - Historical chat tracking

## 🛠️ Setup

1. **Create a Virtual Environment**
bash
python -m venv .env
source .env/bin/activate # On Windows: .env\Scripts\activate

2. **Install Dependencies**
bash
pip install -r requirements.txt

3. **Create Required Directories**
bash
mkdir -p static/images

4. **Add Logo Files**
- Place CN logo at `static/images/cn_logo.png`
- Place Data Science Group logo at `static/images/data_science_group_logo.png`

5. **Configure Environment Variables** (for Gemini AI integration)
tbd

## 🚀 Running the App

bash
streamlit run app.py

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions
├── gemini_helper.py      # Gemini AI integration
├── requirements.txt      # Project dependencies
├── static/              # Static assets
│   └── images/         # Logo files
└── README.md            # This file
```

## 🔧 Configuration

### Styling
- The dashboard uses CN's brand colors:
  - CN Red: #CC0033
  - CN Black: #1A1A1A
  - CN Gray: #666666

### Data Format
The dashboard expects a CSV file with the following columns:
- Date
- Target (historical values)
- Target_Forecast (predicted values)
- Attribution_Indicator_1 through Attribution_Indicator_6 (feature importance scores)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is proprietary and confidential. All rights reserved.

## 🙏 Acknowledgments

- CN Data Science Team
- Streamlit for the amazing framework
- Google for Gemini AI capabilities