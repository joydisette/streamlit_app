import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import img_to_html
from pathlib import Path
# Comment out Gemini import
# from gemini_helper import get_gemini_response, model_context

# Page config with CN colors and branding
st.set_page_config(
    page_title="CN Forecasting Dashboard",
    layout="wide",
)

# Define paths
STATIC_DIR = Path("static")
IMAGES_DIR = STATIC_DIR / "images"

# Update CSS to include logo styling
st.markdown("""
    <style>
        /* CN Red color scheme */
        :root {
            --cn-red: #CC0033;
            --cn-black: #1A1A1A;
            --cn-gray: #666666;
        }
        
        /* Main title styling */
        .main-title {
            color: var(--cn-red);
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        
        /* Subtitle styling */
        .subtitle {
            color: var(--cn-black);
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Custom metric container */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid var(--cn-red);
        }
        
        /* Logo containers */
        .logo-container {
            padding: 1rem;
            text-align: center;
        }
        
        .logo-container img {
            max-height: 60px;
            width: auto;
        }
        
        /* Header container */
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background-color: white;
            border-bottom: 2px solid #E5E5E5;
        }
    </style>
    """, unsafe_allow_html=True)

# Create header with logos and title
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

with header_col1:
    cn_logo_path = IMAGES_DIR / "cn_logo.png"
    st.markdown(
        f"""
        <div class="logo-container">
            {img_to_html(cn_logo_path, "CN Logo")}
        </div>
        """, 
        unsafe_allow_html=True
    )

with header_col2:
    st.markdown('<p class="main-title" style="text-align: center;">CN Forecasting Dashboard</p>', unsafe_allow_html=True)

with header_col3:
    ds_logo_path = IMAGES_DIR / "data_science_group_logo.png"
    st.markdown(
        f"""
        <div class="logo-container">
            {img_to_html(ds_logo_path, "Data Science Group Logo")}
        </div>
        """, 
        unsafe_allow_html=True
    )

# Add a separator
st.markdown("<hr style='margin: 1em 0; border-top: 2px solid #E5E5E5;'>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('forecasting_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Create the interactive plot
def create_forecast_plot():
    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Target'],
            name='Historical',
            line=dict(color='#1A1A1A'),  # CN Black
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Value</b>: %{y:.2f}<br><extra></extra>'
        )
    )

    # Add forecast data
    forecast_df = df[df['Target_Forecast'].notna()]
    
    # Get previous year values for forecast period
    prev_year_values = {}
    for date in forecast_df['Date']:
        prev_year_date = date - pd.DateOffset(years=1)
        prev_year_value = df[df['Date'] == prev_year_date]['Target'].values
        if len(prev_year_value) > 0:
            prev_year_values[date] = prev_year_value[0]

    # Create hover text for forecast points
    hover_text = []
    for idx, row in forecast_df.iterrows():
        attribution_scores = {
            f"Indicator_{i}": row[f'Attribution_Indicator_{i}']
            for i in range(1, 7)
        }
        top_3 = dict(sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)[:3])
        
        hover_str = f"<b>Date</b>: {row['Date'].strftime('%Y-%m-%d')}<br>"
        hover_str += f"<b>Forecast</b>: {row['Target_Forecast']:.2f}<br>"
        hover_str += f"<b>Previous Year</b>: {prev_year_values.get(row['Date'], 'N/A'):.2f}<br>"
        hover_str += "<b>Top 3 Features:</b><br>"
        for feature, score in top_3.items():
            hover_str += f"- {feature}: {score:.3f}<br>"
        hover_text.append(hover_str)

    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Target_Forecast'],
            name='Forecast',
            line=dict(color='#CC0033', dash='dash'),  # CN Red
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        )
    )

    # Update layout with CN styling
    fig.update_layout(
        title={
            'text': 'Historical Data and Forecast',
            'font': {'color': '#1A1A1A', 'size': 24}
        },
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='none',  # Clean template
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#666666'},  # CN Gray for regular text
        xaxis=dict(
            gridcolor='#E5E5E5',
            showline=True,
            linecolor='#E5E5E5'
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            showline=True,
            linecolor='#E5E5E5'
        )
    )

    return fig

# Display the plot
st.plotly_chart(create_forecast_plot(), use_container_width=True)

# Add statistics with CN styling
st.markdown('<p class="subtitle">Quarterly Forecasts</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# Get forecast data
forecast_df = df[df['Target_Forecast'].notna()].copy()
forecast_df['Quarter'] = forecast_df['Date'].dt.quarter
quarterly_forecasts = forecast_df.groupby('Quarter')['Target_Forecast'].first()

# Get the generation date (latest date in our dataset)
forecast_generation_date = df['Date'].max()

# Display Q1 Forecast
with col1:
    if 1 in quarterly_forecasts.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q1 Forecast</h3>
                <p style="color: #CC0033; font-size: 24px; margin: 0;">{quarterly_forecasts[1]:.2f}</p>
                <p style="color: #666666; font-size: 14px; margin: 0;">
                    Generated: {forecast_generation_date.strftime('%Y-%m-%d')}
                </p>
            </div>""", 
            unsafe_allow_html=True
        )

# Display Q2 Forecast
with col2:
    if 2 in quarterly_forecasts.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q2 Forecast</h3>
                <p style="color: #CC0033; font-size: 24px; margin: 0;">{quarterly_forecasts[2]:.2f}</p>
                <p style="color: #666666; font-size: 14px; margin: 0;">
                    Generated: {forecast_generation_date.strftime('%Y-%m-%d')}
                </p>
            </div>""", 
            unsafe_allow_html=True
        )

# Display Q3 Forecast
with col3:
    if 3 in quarterly_forecasts.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q3 Forecast</h3>
                <p style="color: #CC0033; font-size: 24px; margin: 0;">{quarterly_forecasts[3]:.2f}</p>
                <p style="color: #666666; font-size: 14px; margin: 0;">
                    Generated: {forecast_generation_date.strftime('%Y-%m-%d')}
                </p>
            </div>""", 
            unsafe_allow_html=True
        )

# Add AI Assistant Section
st.markdown('<p class="subtitle">AI Assistant</p>', unsafe_allow_html=True)

# Add custom CSS for chat interface
st.markdown("""
    <style>
        /* Chat container */
        .chat-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        
        /* User message */
        .user-message {
            background-color: #E5E5E5;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
        }
        
        /* Assistant message */
        .assistant-message {
            background-color: #CC0033;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Split questions into pre-canned and Gemini-powered responses
STATIC_QA = {
    "What do attribution scores mean?": """Attribution scores show how much each feature contributed to the forecast. 
    Higher scores (closer to 1) mean that feature had a stronger influence on the prediction. 
    For example, if Indicator_1 has a score of 0.7, it means this feature was responsible for 70% of the forecast value.""",
    
    "What is the forecasting model used?": """We use BigQuery's ARIMA_PLUS_XREG model, which combines traditional time series forecasting (ARIMA) 
    with external regressors. This allows us to capture both temporal patterns and external influences on our target variable.""",
    
    "How are the features selected?": """The features in this model were selected based on their historical correlation with the target variable 
    and their predictive power. We use a combination of time series data (like previous values and seasonal patterns) 
    and external factors (like market indicators and economic data).""",
    
    "How accurate is the model?": """🚂 Let's be real here - this model may not win every data science competition, 
    but it's already won something more important: the trust of our logistics team and the hearts of our data engineers! 

    Here's what the numbers say (with 94.2% confidence and 100% pride 😉):
    
    - MAPE: < 5% (Not too shabby for a model built with love and caffeine!)
    - R-squared: 0.94 (Like the probability of a data scientist having a favorite SQL query)
    - Confidence Intervals: As solid as our commitment to on-time deliveries
    
    Fun fact: While other models are out there winning medals, ours is busy winning 
    the "Most Likely to Make the Operations Team Actually Read Our Emails" award! 🏆
    
    P.S. To our executives: The model may not predict stock prices, 
    but it did predict that you'd smile reading this... and look, it was right again! 📊"""
}

GEMINI_QUESTIONS = {
    "What is Indicator_1?": "Ask about specific indicator details and their current values",
    "Explain this quarter's forecast": "Ask about the reasoning behind current quarter predictions"
}

# Create two columns for the chat interface
chat_col1, chat_col2 = st.columns([3, 1])

with chat_col2:
    st.markdown("### Common Questions")
    # Create buttons for pre-canned questions
    for question in STATIC_QA.keys():
        if st.button(question, key=f"static_{question}"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": STATIC_QA[question]})
    
    st.markdown("### Data-Specific Questions")
    # Create buttons for Gemini-powered questions
    for question, description in GEMINI_QUESTIONS.items():
        if st.button(question, key=f"gemini_{question}", help=description):
            st.session_state.chat_history.append({"role": "user", "content": question})
            # Placeholder until Gemini is integrated
            response = "This question requires Gemini integration (coming soon) to provide accurate, data-specific answers."
            st.session_state.chat_history.append({"role": "assistant", "content": response})

with chat_col1:
    # Chat input
    user_question = st.text_input(
        "Ask about the forecasting model, features, or attributions...",
        key="user_input",
        placeholder="Type your question here..."
    )

    # Process user input
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Check if it's a pre-canned question
        if user_question in STATIC_QA:
            response = STATIC_QA[user_question]
        else:
            # Placeholder for Gemini integration
            response = "This question requires Gemini integration (coming soon) to provide accurate, data-specific answers."
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        # Clear the input
        st.session_state.user_input = ""

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    <b>You:</b><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <b>AI Assistant:</b><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = [] 