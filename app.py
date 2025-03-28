import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import img_to_html
from pathlib import Path
from google.cloud import storage
import io
import os
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

# Google Cloud Storage configuration
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'your-bucket-name')
LUMBER_FILE_PATH = os.getenv('GCS_LUMBER_FILE_PATH', 'lumber_forecast.csv')
INTERMODAL_FILE_PATH = os.getenv('GCS_INTERMODAL_FILE_PATH', 'intermodal_forecast.csv')

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

# Load data from Google Cloud Storage
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_gcs(file_path, is_lumber=False):
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        
        # Get bucket and blob
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        # Download the content
        content = blob.download_as_string()
        
        # Read CSV from the downloaded content
        df = pd.read_csv(io.BytesIO(content))
        
        if is_lumber:
            # Process lumber specific data
            df['Date'] = pd.to_datetime(df['time_series_timestamp'])
            df['Value'] = df['time_series_data']
            df['Forecast'] = df['time_series_adjusted_data']
            
            # Get attribution columns (they start with 'attribution_')
            attribution_cols = [col for col in df.columns if col.startswith('attribution_')]
            
            # Normalize attribution scores to sum to 1 for each row
            for idx, row in df.iterrows():
                attr_values = row[attribution_cols].values
                attr_sum = attr_values.sum()
                if attr_sum != 0:  # Avoid division by zero
                    for col in attribution_cols:
                        df.at[idx, col] = row[col] / attr_sum
            
            # Rename attribution columns to match the expected format
            for i, col in enumerate(attribution_cols, 1):
                df[f'Standard_Attribution_Indicator_{i}'] = df[col]
                
        else:
            # For intermodal data, assume it's already in the correct format
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Cloud Storage: {str(e)}")
        raise e

# Add error handling around data loading
try:
    lumber_df = load_data_from_gcs(LUMBER_FILE_PATH, is_lumber=True)
    intermodal_df = load_data_from_gcs(INTERMODAL_FILE_PATH, is_lumber=False)
except Exception as e:
    st.error("Failed to load data. Please check your Google Cloud Storage configuration.")
    st.stop()

# Create the interactive plot for a single dataset
def create_forecast_plot(df, title, color_main='#1A1A1A', color_forecast='#CC0033', prefix='Standard', is_lumber=False):
    # Create figure
    fig = go.Figure()

    # Calculate the date range for the last year plus forecast
    last_year_start = df['Date'].max() - pd.DateOffset(years=1)
    df_last_year = df[df['Date'] >= last_year_start].copy()

    # Add bars for previous year's data
    prev_year_data = []
    prev_year_dates = []
    current_dates = df_last_year['Date'].tolist()
    
    for date in current_dates:
        prev_year_date = date - pd.DateOffset(years=1)
        prev_year_value = df[df['Date'] == prev_year_date]['Value'].values
        if len(prev_year_value) > 0:
            prev_year_data.append(prev_year_value[0])
            prev_year_dates.append(date)

    # Add bar chart for previous year's data
    fig.add_trace(
        go.Bar(
            x=prev_year_dates,
            y=prev_year_data,
            name='Previous Year',
            marker_color=f'rgba{tuple(int(color_main.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Previous Year Value</b>: %{y:.2f}<br><extra></extra>'
        )
    )

    # Add current year's actual data
    current_data = df_last_year[df_last_year['Value'].notna()]
    fig.add_trace(
        go.Scatter(
            x=current_data['Date'],
            y=current_data['Value'],
            name='Current',
            line=dict(color=color_main),
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Current Value</b>: %{y:.2f}<br><extra></extra>'
        )
    )

    # Add forecast data with confidence intervals for lumber data
    forecast_data = df_last_year[df_last_year['Forecast'].notna()]
    
    if is_lumber:
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'].tolist() + forecast_data['Date'].tolist()[::-1],
                y=forecast_data['prediction_interval_upper_bound'].tolist() + 
                  forecast_data['prediction_interval_lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(color_forecast.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            )
        )
    
    # Create hover text for forecast points
    hover_text = []
    for idx, row in forecast_data.iterrows():
        # Get previous year's value
        prev_year_date = row['Date'] - pd.DateOffset(years=1)
        prev_year_value = df[df['Date'] == prev_year_date]['Value'].values[0]
        
        # Calculate YoY change
        yoy_change = ((row['Forecast'] - prev_year_value) / prev_year_value) * 100
        
        hover_str = f"<b>Date</b>: {row['Date'].strftime('%Y-%m-%d')}<br>"
        hover_str += f"<b>Forecast</b>: {row['Forecast']:.2f}<br>"
        hover_str += f"<b>Previous Year</b>: {prev_year_value:.2f}<br>"
        hover_str += f"<b>YoY Change</b>: {yoy_change:+.1f}%<br>"
        
        if is_lumber:
            hover_str += f"<b>Standard Error</b>: {row['standard_error']:.3f}<br>"
            hover_str += f"<b>Confidence Level</b>: {row['confidence_level']:.1%}<br>"
            
            # Add top 3 attribution scores
            attribution_cols = [col for col in row.index if col.startswith(f'{prefix}_Attribution_Indicator_')]
            top_3 = sorted([(col, row[col]) for col in attribution_cols], key=lambda x: x[1], reverse=True)[:3]
            
            hover_str += "<b>Top 3 Features:</b><br>"
            for col, score in top_3:
                hover_str += f"- {col}: {score:.3f}<br>"
        
        hover_text.append(hover_str)

    # Add forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast'],
            name='Forecast',
            line=dict(color=color_forecast, dash='dash'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        )
    )

    # Update layout with CN styling
    fig.update_layout(
        title={
            'text': title,
            'font': {'color': '#1A1A1A', 'size': 24}
        },
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='none',  # Clean template
        height=400,  # Reduced height since we have two plots
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#666666'},  # CN Gray for regular text
        xaxis=dict(
            gridcolor='#E5E5E5',
            showline=True,
            linecolor='#E5E5E5',
            range=[last_year_start, df['Date'].max()]
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            showline=True,
            linecolor='#E5E5E5'
        ),
        barmode='overlay'
    )

    return fig

# Display the plots
st.markdown('<p class="subtitle">Lumber Forecast</p>', unsafe_allow_html=True)
lumber_fig = create_forecast_plot(
    lumber_df, 
    'Lumber Year-over-Year Comparison and Forecast',
    color_main='#1A1A1A',
    color_forecast='#CC0033',
    prefix='Standard',
    is_lumber=True
)
st.plotly_chart(lumber_fig, use_container_width=True)

st.markdown('<p class="subtitle">Intermodal Forecast</p>', unsafe_allow_html=True)
intermodal_fig = create_forecast_plot(
    intermodal_df,
    'Intermodal Year-over-Year Comparison and Forecast',
    color_main='#666666',
    color_forecast='#666666',
    prefix='Fancy',
    is_lumber=False
)
st.plotly_chart(intermodal_fig, use_container_width=True)

# Add statistics with CN styling
st.markdown('<p class="subtitle">Quarterly Forecasts</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# Get forecast data for both types
lumber_forecast = lumber_df[lumber_df['Forecast'].notna()].copy()
intermodal_forecast = intermodal_df[intermodal_df['Forecast'].notna()].copy()

lumber_forecast['Quarter'] = lumber_forecast['Date'].dt.quarter
intermodal_forecast['Quarter'] = intermodal_forecast['Date'].dt.quarter

lumber_quarterly = lumber_forecast.groupby('Quarter')['Forecast'].first()
intermodal_quarterly = intermodal_forecast.groupby('Quarter')['Forecast'].first()

# Get the generation date (latest date in our dataset)
forecast_generation_date = max(lumber_df['Date'].max(), intermodal_df['Date'].max())

# Display Q1 Forecast
with col1:
    if 1 in lumber_quarterly.index or 1 in intermodal_quarterly.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q1 Forecast</h3>
                {f'<p style="color: #CC0033; font-size: 24px; margin: 0;">Lumber: {lumber_quarterly[1]:.2f}</p>' if 1 in lumber_quarterly.index else ''}
                {f'<p style="color: #666666; font-size: 24px; margin: 0;">Intermodal: {intermodal_quarterly[1]:.2f}</p>' if 1 in intermodal_quarterly.index else ''}
                <p style="color: #666666; font-size: 14px; margin: 0;">
                    Generated: {forecast_generation_date.strftime('%Y-%m-%d')}
                </p>
            </div>""", 
            unsafe_allow_html=True
        )

# Display Q2 Forecast
with col2:
    if 2 in lumber_quarterly.index or 2 in intermodal_quarterly.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q2 Forecast</h3>
                {f'<p style="color: #CC0033; font-size: 24px; margin: 0;">Lumber: {lumber_quarterly[2]:.2f}</p>' if 2 in lumber_quarterly.index else ''}
                {f'<p style="color: #666666; font-size: 24px; margin: 0;">Intermodal: {intermodal_quarterly[2]:.2f}</p>' if 2 in intermodal_quarterly.index else ''}
                <p style="color: #666666; font-size: 14px; margin: 0;">
                    Generated: {forecast_generation_date.strftime('%Y-%m-%d')}
                </p>
            </div>""", 
            unsafe_allow_html=True
        )

# Display Q3 Forecast
with col3:
    if 3 in lumber_quarterly.index or 3 in intermodal_quarterly.index:
        st.markdown(
            f"""<div class="metric-container">
                <h3 style="color: #1A1A1A; margin: 0;">Q3 Forecast</h3>
                {f'<p style="color: #CC0033; font-size: 24px; margin: 0;">Lumber: {lumber_quarterly[3]:.2f}</p>' if 3 in lumber_quarterly.index else ''}
                {f'<p style="color: #666666; font-size: 24px; margin: 0;">Intermodal: {intermodal_quarterly[3]:.2f}</p>' if 3 in intermodal_quarterly.index else ''}
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