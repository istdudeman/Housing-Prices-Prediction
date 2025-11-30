"""
Streamlit App for Housing Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    /* Remove background from metric boxes */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    h1 {
        color: #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model configuration
@st.cache_resource
def load_model_config():
    try:
        with open('model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error("⚠️ Model configuration file not found. Please run train_and_export_model.py first.")
        st.stop()

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('housing_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.error("Please run train_and_export_model.py first to train and save the model.")
        st.stop()

# Load everything
config = load_model_config()
model, scaler, feature_names = load_model_and_scaler()

# Title and description
st.title("🏠 Housing Price Prediction System")
st.markdown("### Predict house prices using machine learning")

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    
    model_info = config['model_info']
    st.markdown(f"**Model:** {model_info['name']}")
    st.markdown(f"**Type:** {model_info['type']}")
    st.markdown(f"**Version:** {model_info['version']}")
    st.markdown(f"**Created:** {model_info['created_date']}")
    
    st.divider()
    
    st.header("📈 Performance Metrics")
    metrics = config['performance_metrics']
    st.metric("R² Score", f"{metrics['r2_score']:.4f}")
    st.metric("RMSE", f"${metrics['rmse']:,.2f}")
    st.metric("MAE", f"${metrics['mae']:,.2f}")
    st.metric("Accuracy", f"{metrics['accuracy_percentage']:.2f}%")
    
    st.divider()
    
    st.header("ℹ️ Data Info")
    data_info = config['data_info']
    st.markdown(f"**Training Samples:** {data_info['training_samples']:,}")
    st.markdown(f"**Total Features:** {data_info['total_features']}")
    st.markdown(f"**Target:** {data_info['target_column']}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Feature Importance", "📋 Model Details", "📁 Upload Data"])

# Tab 1: Prediction
with tab1:
    st.header("Make a Prediction")
    st.markdown("Upload a CSV file with housing features or enter values manually")
    
    # Option to upload CSV
    uploaded_file = st.file_uploader("Upload CSV file with housing data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            input_data = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(input_data)} records")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                st.dataframe(input_data.head())
            
            if st.button("🔮 Predict Prices", type="primary"):
                with st.spinner("Making predictions..."):
                    # Preprocess the data (same as training)
                    # This is a simplified version - you may need to adjust based on your actual preprocessing
                    X_encoded = pd.get_dummies(input_data)
                    
                    # Align columns with training data
                    missing_cols = set(feature_names) - set(X_encoded.columns)
                    for col in missing_cols:
                        X_encoded[col] = 0
                    
                    # Ensure column order matches
                    X_encoded = X_encoded[feature_names]
                    
                    # Fill missing values
                    X_encoded = X_encoded.fillna(X_encoded.median())
                    
                    # Scale and predict
                    X_scaled = scaler.transform(X_encoded)
                    predictions = model.predict(X_scaled)
                    
                    # Add predictions to dataframe
                    result_df = input_data.copy()
                    result_df['Predicted_Price'] = predictions
                    
                    # Display results
                    st.success("✓ Predictions completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Price", f"${predictions.mean():,.2f}")
                    with col2:
                        st.metric("Min Price", f"${predictions.min():,.2f}")
                    with col3:
                        st.metric("Max Price", f"${predictions.max():,.2f}")
                    
                    # Show results
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    if len(predictions) <= 100:
                        fig = px.bar(
                            x=range(len(predictions)),
                            y=predictions,
                            labels={'x': 'House Index', 'y': 'Predicted Price ($)'},
                            title='Predicted Prices Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(
                            predictions,
                            nbins=50,
                            labels={'value': 'Predicted Price ($)', 'count': 'Frequency'},
                            title='Predicted Prices Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Make sure your CSV file has the same columns as the training data.")
    else:
<<<<<<< Updated upstream
        st.info("👆 Upload a CSV file to get started with batch predictions")
        st.markdown("---")
        st.markdown("### Quick Single Prediction")
        st.markdown("For single predictions, you would need to create input fields for all features.")
        st.markdown(f"**Total features required:** {len(feature_names)}")
=======
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: #e8f4f8; border-radius: 10px; margin: 2rem 0; border: 2px dashed #1f77b4;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>👆 Upload file CSV untuk memulai prediksi</h3>
            <p style='color: #34495e; font-size: 1.1rem;'>Pilih file CSV yang berisi data apartemen yang ingin diprediksi harganya</p>
        </div>
        """, unsafe_allow_html=True)
>>>>>>> Stashed changes
        
        with st.expander("View required features"):
            st.write(feature_names)

# Tab 2: Feature Importance
with tab2:
    st.header("Feature Importance Analysis")
    
    feature_importance = config['feature_importance']
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    # Plot
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 20 Most Important Features',
        labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table
    st.subheader("Feature Importance Table")
    importance_table = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_table.index += 1
    st.dataframe(importance_table, use_container_width=True)

# Tab 3: Model Details
with tab3:
    st.header("Model Configuration Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        hyperparams = config['hyperparameters']
        for param, value in hyperparams.items():
            st.markdown(f"**{param}:** `{value}`")
    
    with col2:
        st.subheader("Preprocessing")
        preprocessing = config['preprocessing']
        for step, method in preprocessing.items():
            st.markdown(f"**{step}:** {method}")
    
    st.divider()
    
    st.subheader("Complete YAML Configuration")
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language='yaml')
    
    # Download YAML
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    st.download_button(
        label="📥 Download YAML Config",
        data=yaml_str,
        file_name="model_config.yaml",
        mime="text/yaml"
    )

# Tab 4: Upload Data for Prediction
with tab4:
    st.header("Test Data Upload")
    st.markdown("Upload your test dataset to generate predictions")
    
    # Check if test file exists
    try:
        test_data = pd.read_csv('test_100k.csv')
        st.success(f"✓ Found test_100k.csv with {len(test_data)} records")
        
        with st.expander("Preview test data"):
            st.dataframe(test_data.head(10))
        
        if st.button("🔮 Predict on Test Data", type="primary"):
            with st.spinner("Processing test data..."):
                # Preprocess
                X_test_encoded = pd.get_dummies(test_data)
                
                # Align columns
                missing_cols = set(feature_names) - set(X_test_encoded.columns)
                for col in missing_cols:
                    X_test_encoded[col] = 0
                
                X_test_encoded = X_test_encoded[feature_names]
                X_test_encoded = X_test_encoded.fillna(X_test_encoded.median())
                
                # Predict
                X_test_scaled = scaler.transform(X_test_encoded)
                predictions = model.predict(X_test_scaled)
                
                # Create results
                results = pd.DataFrame({
                    'Id': range(len(predictions)),
                    'Predicted_Price': predictions
                })
                
                st.success("✓ Predictions completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", f"{len(predictions):,}")
                with col2:
                    st.metric("Average Price", f"${predictions.mean():,.2f}")
                with col3:
                    st.metric("Median Price", f"${np.median(predictions):,.2f}")
                
                st.dataframe(results.head(20))
                
                # Download
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Predictions",
                    data=csv,
                    file_name="housing_price_predictions.csv",
                    mime="text/csv"
                )
                
    except FileNotFoundError:
        st.info("No test_100k.csv file found in the current directory")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Housing Price Prediction System | Powered by Random Forest & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
