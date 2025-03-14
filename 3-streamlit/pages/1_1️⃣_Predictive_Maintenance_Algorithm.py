import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt



st.set_page_config(page_title="Predictive Maintenance Algorithm")

# Title & Description
st.title("🔧 Predictive Maintenance Algorithm")
st.markdown("---")
st.write("")
st.markdown("""
Provide sensor data inputs manually or upload your CSV to predict machine failure modes
""")

# Sidebar
st.sidebar.header("⚙️ Prediction Settings")
st.sidebar.info("Using Gradient Boosting Model (optimized & tuned).")

# User input form
with st.expander("📝 Enter Machine Parameters Manually"):
    air_temp = st.number_input("Air Temperature [K]", value=300, step=1)
    process_temp = st.number_input("Process Temperature [K]", value=310, step=1)
    rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500, step=10)
    torque = st.number_input("Torque [Nm]", value=40, step=1)
    tool_wear = st.number_input("Tool Wear [min]", value=100, step=1)

# File uploader for batch predictions
st.write("")
uploaded_file = st.file_uploader("📂 Or upload a CSV file for batch predictions", type=["csv"])

# Load trained model
model_path = Path(__file__).resolve().parents[2] / "1-algorithm" / "best_gradient_boosting_model.pkl"
model = joblib.load(model_path)


if st.button("🚀 Predict Failure Mode"):

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.DataFrame({
            "Air_temperature": [air_temp],
            "Process_temperature": [process_temp],
            "Rotational_speed": [rotational_speed],
            "Torque": [torque],
            "Tool_wear": [tool_wear]
        })

    # Run prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Use the model's built-in classes directly for robustness
    # failure_types = model.classes_
    failure_types = [
        "No Failure",
        "Tool Wear (TWF)",
        "Heat Dissipation (HDF)",
        "Power (PWF)",
        "Overstrain (OSF)"
    ]

    # Map predictions to labels
    predicted_failure = [failure_types[p] for p in prediction]

    # Display ONLY the final failure result
    st.write("")
    st.write("")
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if predicted_failure[0] == "No Failure":
        st.success("✅ Machine is operating normally. No imminent failures detected.")
    else:
        st.write("")
        st.error(f"⚠️ Predicted Failure: {predicted_failure[0]}")

        proba_df = pd.DataFrame(prediction_proba, columns=failure_types).T
        proba_df.columns = ["Probability"]
        plot_df = proba_df.reset_index().rename(columns={'index': 'Failure Mode'})

        # horizontal bar chart
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                y=alt.Y("Failure Mode", sort=None),
                x=alt.X("Probability", scale=alt.Scale(domain=[0,1])), 
                tooltip=["Failure Mode", "Probability"]
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)


        # Chatbot integration
        st.session_state["predicted_failure"] = predicted_failure[0]

        st.markdown("""
            <a href="/Chatbot" target="_self">
                <button style='
                    background-color:#FFFFFF;
                    color:black;
                    padding: 10px;
                    border-radius:8px;
                    border:none;
                    cursor:pointer;
                '>💬 Ask Chatbot How to Solve This Issue</button>
            </a>
            """,
            unsafe_allow_html=True
        )