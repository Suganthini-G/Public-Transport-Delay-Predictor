# 🚌 Sri Lanka Public Transport Delay Predictor

An interactive machine learning--powered web application that predicts
the likelihood of delays in Sri Lanka's public transport system based on
journey details, traffic conditions, weather, and time-related factors.

------------------------------------------------------------------------

## 🛠️ Installation

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Run the Streamlit Application

``` bash
streamlit run Public_Transport_Delay_Predictor.py
```

### Open in Browser

Navigate to:

    http://localhost:8501

------------------------------------------------------------------------

## 📖 How to Use

### Enter Journey Details

-   Select transport mode, route type, and distance\
-   Choose time of day and day type\
-   Specify traffic level, weather condition, and special events

### Predict Delay Risk

-   Click **"Predict Delay Risk"**
-   The system analyzes all inputs using a trained machine learning
    model

### View Results

-   Delay risk classification (**High / Low**)\
-   Delay probability percentage\
-   Visual gauge and confidence breakdown\
-   Practical travel recommendations

------------------------------------------------------------------------

## 🧠 Machine Learning Approach

-   **Problem Type**: Binary Classification\
    *(Delay ≥ 10 minutes vs No Significant Delay)*

-   **Primary Model**: XGBoost Classifier

### Supporting Components

-   Feature scaling using **StandardScaler**
-   Label encoding for categorical variables
-   Threshold-based classification decision

### Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   ROC-AUC\
-   Balanced Accuracy

------------------------------------------------------------------------

## 📊 Model Inputs

### Core Features

-   Transport mode\
-   Route type\
-   Distance (km)\
-   Time of day\
-   Day type\
-   Traffic level\
-   Weather condition\
-   Special events

### Engineered Features

-   Peak hour indicator\
-   High traffic flag\
-   Bad weather flag\
-   Weekend indicator\
-   Long-distance travel\
-   Interaction-based features

------------------------------------------------------------------------

## 🎨 User Interface Highlights

-   Gradient-based modern design\
-   Responsive wide-screen layout\
-   Styled input components and expanders\
-   Interactive Plotly visualizations\
-   Clear warning and success indicators

------------------------------------------------------------------------

## 📝 Project Structure

    Public-Transport-Delay-Predictor/
    │
    ├── Public_Transport_Delay_Predictor.py
    ├── requirements.txt
    ├── runtime.txt
    ├── models/
    │   └── best_model.pkl
    └── README.md

------------------------------------------------------------------------

## 🚀 Deployment

Designed for **Streamlit Cloud** deployment.

### Requirements

-   `requirements.txt` for dependencies\
-   `runtime.txt` to lock Python version (**3.10**)\
-   Pre-trained model file included
-   
------------------------------------------------------------------------
