import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Sri Lanka Transport Delay Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main title styling */
    html, body, .stApp {
        background-color: #ffffff !important;
        color: #000000;
    }

    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Feature importance card */
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
            
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
            
    [data-testid="stExpander"] details summary {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }
    
    [data-testid="stExpander"] details summary:hover {
        background-color: #e2e8f0 !important;
        border-color: #667eea !important;
    }
    
    [data-testid="stExpander"] details[open] summary {
        background-color: #e2e8f0 !important;
        border-radius: 10px 10px 0 0 !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 24px !important;
    }
    
    /* Input Labels - Dark & Bold */
    .stSelectbox label, 
    .stSlider label, 
    .stTextInput label,
    .stNumberInput label {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 8px !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        padding-top: 12px !important;
    }
    
    .stSlider [data-testid="stTickBar"] {
        color: #1e293b !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }

    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [role="listbox"] {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    [role="option"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        padding: 12px 16px !important;
    }
    
    [role="option"]:hover {
        background-color: #f1f5f9 !important;
        color: #667eea !important;
    }
    
    [role="option"][aria-selected="true"] {
        background-color: #e0e7ff !important;
        color: #4338ca !important;
        font-weight: 600 !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        padding-top: 12px !important;
    }
    
    .stSlider [data-testid="stTickBar"] {
        color: #1e293b !important;
    }
    
    /* Horizontal Rules - Visible */
    hr {
        border: none !important;
        height: 3px !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        margin: 2.5rem 0 !important;
        opacity: 1 !important;
        border-radius: 2px !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p {
        color: #1e293b !important;
    }
    
    .stAlert {
        color: #1e293b !important; 
    }
    
    .stAlert p, .stAlert ul, .stAlert li {
        color: #1e293b !important;  
        font-weight: 500 !important;
    }
    
    .stAlert strong {
        color: #0f172a !important;  
        font-weight: 700 !important;
    }
    
    /* Specific for warning boxes (yellow) */
    [data-testid="stWarning"] {
        color: #854d0e !important;  
    }
    
    [data-testid="stWarning"] p,
    [data-testid="stWarning"] ul,
    [data-testid="stWarning"] li {
        color: #854d0e !important;
    }
    
    /* Specific for success boxes (green) */
    [data-testid="stSuccess"] {
        color: #065f46 !important;  
    }
    
    [data-testid="stSuccess"] p,
    [data-testid="stSuccess"] ul,
    [data-testid="stSuccess"] li {
        color: #065f46 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model_bundle = load_model()

model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoders = model_bundle["label_encoders"]
feature_names = model_bundle["feature_names"]

threshold = 0.5


st.markdown('''
    <h1 style="text-align: center; font-size: 3rem; font-weight: bold; padding: 1rem; margin-bottom: 0.5rem;">
    <span style="font-size: 3.5rem;">🚌</span>
    <span class="main-title" style="display: inline;">Sri Lanka Public Transport Delay Predictor</span>
    </h1>
    ''', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Delays Before You Travel</p>', unsafe_allow_html=True)

st.markdown("---")

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Enter Journey Details")

    with st.expander("🚍 Transport Information", expanded=True):
        mode = st.selectbox(
            "Transport Mode",
            ["Bus", "Train"],
            help="Select your mode of transport"
        )
        
        route = st.selectbox(
            "Route Type",
            ["Urban", "Intercity"],
            help="Urban routes are within city limits, Intercity routes connect different cities"
        )
        
        distance_km = st.slider(
            "Journey Distance (km)",
            min_value=1.0,
            max_value=200.0,
            value=15.0,
            step=0.5,
            help="Total distance of your journey"
        )
    
    with st.expander("⏰ Timing & Schedule", expanded=True):
        time = st.selectbox(
            "Time of Day",
            ["Morning", "Afternoon", "Evening", "Night"],
            help="Morning: 6AM-12PM | Afternoon: 12PM-5PM | Evening: 5PM-9PM | Night: 9PM-6AM"
        )
        
        day_type = st.selectbox(
            "Day Type",
            ["Weekday", "Weekend"],
            help="Weekends typically have different traffic patterns"
        )
    
    with st.expander("🌦️ Conditions", expanded=True):
        traffic = st.selectbox(
            "Current Traffic Level",
            ["Low", "Medium", "High"],
            help="Check real-time traffic conditions before selecting"
        )
        
        weather = st.selectbox(
            "Weather Condition",
            ["Clear", "Rainy", "Heavy Rain"],
            help="Current weather conditions affect delay probability"
        )
        
        event = st.selectbox(
            "Special Event Nearby?",
            ["No", "Yes"],
            help="Concerts, sports events, festivals can cause delays"
        )
    
    st.markdown("---")
    st.markdown("### 📝 Journey Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.info(f"""
        **🚌 Vehicle:** {mode}  
        **📍 Route:** {route}  
        **📏 Distance:** {distance_km} km
        """)
    
    with summary_col2:
        st.info(f"""
        **⏰ Time:** {time}  
        **📅 Day:** {day_type}  
        **🌤️ Weather:** {weather}
        """)
    
    is_peak_hour = time in ["Morning", "Evening"]
    high_traffic = traffic == "High"
    bad_weather = weather in ["Rainy", "Heavy Rain"]
    event_impact = (event == "Yes") and (traffic != "Low")
    is_weekend = day_type == "Weekend"
    long_distance = distance_km >= 15
    rush_bad_weather = is_peak_hour and bad_weather
    bus_intercity = (mode == "Bus") and (route == "Intercity")
    event_high_traffic = (event == "Yes") and (traffic == "High")
    
    if distance_km <= 10:
        distance_category = "Short"
    elif distance_km <= 30:
        distance_category = "Medium"
    elif distance_km <= 100:
        distance_category = "Long"
    else:
        distance_category = "Very_Long"
    
    distance_normalized = (distance_km - 1.0) / (200.0 - 1.0)
    
    input_df = pd.DataFrame([{
        "transport_mode": mode,
        "time_of_day": time,
        "traffic_level": traffic,
        "weather": weather,
        "route_type": route,
        "special_event": event,
        "day_type": day_type,
        "distance_km": distance_km,
        "is_peak_hour": int(is_peak_hour),
        "high_traffic": int(high_traffic),
        "bad_weather": int(bad_weather),
        "event_impact": int(event_impact),
        "is_weekend": int(is_weekend),
        "long_distance": int(long_distance),
        "event_high_traffic": int(event_high_traffic),
        "bus_intercity": int(bus_intercity),
        "distance_category": distance_category,
        "rush_bad_weather": int(rush_bad_weather),
        "distance_normalized": distance_normalized
    }])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Delay Risk", use_container_width=True)

with col_result:
    st.markdown("### Prediction Results")
    
    if predict_button:
        with st.spinner("🔄 Analyzing journey conditions..."):
            numeric_features = [
                "distance_km",
                "is_peak_hour",
                "high_traffic",
                "bad_weather",
                "is_weekend",
                "long_distance",
                "event_high_traffic",
                "bus_intercity",
                "rush_bad_weather",
                "distance_normalized"
            ]

            for col, encoder in label_encoders.items():
                input_df[col] = encoder.transform(input_df[col])

            input_df[numeric_features] = scaler.transform(input_df[numeric_features])

            input_df = input_df[feature_names]

            delay_probability = model.predict_proba(input_df)[0][1]
            prediction = delay_probability >= threshold
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction:
                st.markdown(f"""
                    <div class="warning-card">
                        <h2 style="margin:0;">⚠️ HIGH DELAY RISK</h2>
                        <h1 style="margin:0.5rem 0; font-size: 3rem;">{delay_probability:.1%}</h1>
                        <p style="margin:0; font-size: 1.1rem;">Your journey is likely to be delayed by ≥10 minutes</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.warning("**💡 Recommendations:**")
                st.markdown("""
                - 🕐 Consider leaving **15-20 minutes earlier**
                - 🚗 Check alternative routes or transport modes
                - 📱 Monitor real-time traffic updates
                - ⏰ Plan buffer time for important appointments
                """)
            else:
                st.markdown(f"""
                    <div class="success-card">
                        <h2 style="margin:0;">✅ LOW DELAY RISK</h2>
                        <h1 style="margin:0.5rem 0; font-size: 3rem;">{delay_probability:.1%}</h1>
                        <p style="margin:0; font-size: 1.1rem;">Your journey should proceed on schedule</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success("**💡 Journey Tips:**")
                st.markdown("""
                - ✅ Journey conditions look favorable
                - 🎫 Arrive at stop/station 5 minutes early
                - 📱 Still recommended to check live updates
                - 😊 Have a pleasant journey!
                """)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Delay Probability Gauge")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=delay_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Delay Risk (%)", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#00f2fe'},
                        {'range': [30, 70], 'color': '#ffd700'},
                        {'range': [70, 100], 'color': '#f5576c'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("#### 🤖 Model Confidence Breakdown")
            
            confidence_data = pd.DataFrame({
                'Model': ['XGBoost'],
                'Probability': [delay_probability * 100],
                'Weight': [100]
            })
            
            fig_conf = px.bar(
                confidence_data,
                x='Model',
                y='Probability',
                title='Individual Model Predictions',
                color='Probability',
                color_continuous_scale=['#00f2fe', '#ffd700', '#f5576c'],
                text='Probability'
            )
            
            fig_conf.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                textfont=dict(size=18, color='black', family='Arial Black')
            )
            
            fig_conf.update_layout(
                showlegend=False,
                height=300,
                yaxis_range=[0, 100],
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black", size=14),
                title=dict(text='Individual Model Predictions', font=dict(color='#1e293b', size=18, family='Arial Bold')),
                xaxis=dict(
                    title=dict(text='Model', font=dict(color='black', size=14)),
                    tickfont=dict(color='black', size=13),
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text='Delay Probability (%)', font=dict(color='black', size=14)),
                    tickfont=dict(color='black', size=13),
                    gridcolor="#e5e7eb"
                ),
                coloraxis_colorbar=dict(         
                    title=dict(text="Probability", font=dict(color='#000000', size=14, family='Arial Bold')),         
                    tickfont=dict(color='#000000', size=12)
                )
            )

            
            st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("👈 Enter your journey details and click 'Predict Delay Risk' to see results")
        
        st.markdown("#### How It Works")
        
        st.markdown("""
        We analyzes multiple factors to predict delay probability:
        
        1. **🚌 Transport Details** - Mode, route type, distance
        2. **⏰ Timing Factors** - Time of day, day type, peak hours
        3. **🌦️ Environmental Conditions** - Traffic, weather
        4. **🎪 Special Events** - Concerts, festivals, holidays
        5. **🔄 Feature Interactions** - Combined effects
        """)


st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Sri Lanka Public Transport Delay Predictor</strong></p>
    </div>
""", unsafe_allow_html=True)
