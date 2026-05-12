import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Universal AI Analytics Platform",
    page_icon="🚀",
    layout="wide"
)

# ================= DATASET TYPE DETECTION =================

def detect_dataset_type(columns):

    columns = [str(col).lower() for col in columns]

    if any(word in columns for word in [
        "fraud", "transaction", "merchant", "amount"
    ]):
        return "Fraud Detection Dataset"

    elif any(word in columns for word in [
        "employee", "salary", "department", "attrition"
    ]):
        return "HR Analytics Dataset"

    elif any(word in columns for word in [
        "disease", "patient", "diagnosis", "medical"
    ]):
        return "Healthcare Dataset"

    elif any(word in columns for word in [
        "sales", "revenue", "profit", "product"
    ]):
        return "Sales Analytics Dataset"

    elif any(word in columns for word in [
        "churn", "customer", "tenure"
    ]):
        return "Customer Churn Dataset"

    else:
        return "Generic Analytics Dataset"

# ================= PDF REPORT =================

def generate_pdf_report(dataset_type, rows, cols):

    doc = SimpleDocTemplate("analytics_report.pdf")

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "AI Analytics Platform Report",
        styles['Title']
    )

    content.append(title)

    text = Paragraph(
        f"""
        Dataset Type: {dataset_type}<br/>
        Number of Rows: {rows}<br/>
        Number of Columns: {cols}<br/>
        AI analytics completed successfully.
        """,
        styles['BodyText']
    )

    content.append(text)

    doc.build(content)

# ================= TITLE =================

st.title("🚀 Universal AI Analytics SaaS Platform")

st.markdown("""
Upload ANY dataset and automatically:
- analyze data
- train AI model
- generate dashboards
- make predictions
- export PDF reports
""")

# ================= FILE UPLOAD =================

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ================= MAIN APP =================

if uploaded_file is not None:

    try:

        # ================= LOAD DATASET =================

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # ================= DATASET TYPE =================

        dataset_type = detect_dataset_type(df.columns)

        st.info(f"Detected Dataset Type: {dataset_type}")

        # ================= BASIC INFO =================

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        # ================= TARGET COLUMN =================

        target_column = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # ================= DATA CLEANING =================

        clean_df = df.copy()

        label_encoders = {}

        for column in clean_df.columns:

            try:

                clean_df[column] = pd.to_numeric(
                    clean_df[column]
                )

                clean_df[column] = clean_df[column].fillna(
                    clean_df[column].median()
                )

            except:

                clean_df[column] = clean_df[column].astype(str)

                clean_df[column] = clean_df[column].fillna(
                    "Unknown"
                )

                le = LabelEncoder()

                clean_df[column] = le.fit_transform(
                    clean_df[column]
                )

                label_encoders[column] = le

        # ================= ANALYTICS =================

        st.subheader("Dataset Analytics")

        fig = px.histogram(
            clean_df,
            x=target_column,
            title="Target Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        numeric_df = clean_df.select_dtypes(
            include=np.number
        )

        corr = numeric_df.corr()

        heatmap = px.imshow(
            corr,
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

        # ================= FEATURE COLUMNS =================

        feature_columns = [
            col for col in clean_df.columns
            if col != target_column
        ]

        # ================= SESSION STATE =================

        if "sample_data" not in st.session_state:

            st.session_state["sample_data"] = {
                col: 0
                for col in feature_columns
            }

        # ================= RANDOM SAMPLE =================

        st.subheader("🎲 Random Dataset Sample")

        colA, colB = st.columns(2)

        with colA:

            if st.button("Generate Random Dataset Values"):

                random_row = df.sample(
                    n=1
                ).iloc[0]

                st.session_state["sample_data"] = {
                    col: random_row[col]
                    for col in feature_columns
                }

                st.success(
                    "Random values generated!"
                )

                st.rerun()

        with colB:

            if st.button("Reset All Values"):

                st.session_state["sample_data"] = {
                    col: 0
                    for col in feature_columns
                }

                st.rerun()

        # ================= MODEL TRAINING =================

        st.subheader("🤖 AI Model Training")

        if st.button("Train AI Model"):

            with st.spinner(
                "Training AI Model... Please wait..."
            ):

                # Faster training for huge datasets
                if len(clean_df) > 50000:

                    clean_df = clean_df.sample(
                        50000,
                        random_state=42
                    )

                X = clean_df.drop(
                    target_column,
                    axis=1
                )

                y = clean_df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                model = RandomForestClassifier(
                    n_estimators=20,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                st.success(
                    f"Model Trained Successfully! Accuracy: {accuracy:.2%}"
                )

                # ================= SAVE MODEL =================

                os.makedirs(
                    "models",
                    exist_ok=True
                )

                joblib.dump(
                    model,
                    "models/dynamic_model.pkl"
                )

                joblib.dump(
                    X.columns.tolist(),
                    "models/columns.pkl"
                )

                joblib.dump(
                    label_encoders,
                    "models/encoders.pkl"
                )

                st.success(
                    "Model Saved Successfully!"
                )

                # ================= PDF REPORT =================

                generate_pdf_report(
                    dataset_type,
                    df.shape[0],
                    df.shape[1]
                )

                with open(
                    "analytics_report.pdf",
                    "rb"
                ) as pdf_file:

                    st.download_button(
                        "Download PDF Report",
                        pdf_file,
                        file_name="analytics_report.pdf",
                        mime="application/pdf"
                    )

                # ================= FEATURE IMPORTANCE =================

                importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                })

                importance_df = importance_df.sort_values(
                    by="Importance",
                    ascending=False
                )

                fig2 = px.bar(
                    importance_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importance"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        # ================= PREDICTION =================

        st.subheader("🔮 Real-Time Prediction")

        input_data = {}

        for column in feature_columns:

            original_column = df[column]

            # NUMERIC COLUMN
            if pd.api.types.is_numeric_dtype(
                original_column
            ):

                default_value = st.session_state[
                    "sample_data"
                ].get(column, 0)

                input_data[column] = st.number_input(
                    f"{column}",
                    value=float(default_value),
                    key=column
                )

            # CATEGORICAL COLUMN
            else:

                unique_values = original_column.astype(
                    str
                ).unique().tolist()

                default_value = str(
                    st.session_state[
                        "sample_data"
                    ].get(column, unique_values[0])
                )

                if default_value not in unique_values:

                    default_value = unique_values[0]

                input_data[column] = st.selectbox(
                    f"{column}",
                    options=unique_values,
                    index=unique_values.index(
                        default_value
                    )
                )

        # ================= PREDICT =================

        if st.button("Predict"):

            if not os.path.exists(
                "models/dynamic_model.pkl"
            ):

                st.error(
                    "Please train the AI model first!"
                )

            else:

                model = joblib.load(
                    "models/dynamic_model.pkl"
                )

                expected_columns = joblib.load(
                    "models/columns.pkl"
                )

                label_encoders = joblib.load(
                    "models/encoders.pkl"
                )

                input_df = pd.DataFrame(
                    [input_data]
                )

                # Encode categorical inputs
                for column in input_df.columns:

                    if column in label_encoders:

                        le = label_encoders[column]

                        input_df[column] = le.transform(
                            input_df[column].astype(str)
                        )

                input_df = input_df.reindex(
                    columns=expected_columns
                )

                input_df = input_df.fillna(0)

                prediction = model.predict(
                    input_df
                )[0]

                st.success(
                    f"Prediction Result: {prediction}"
                )

    except Exception as e:

        st.error(f"Error: {str(e)}")

else:

    st.info(
        "Please upload a CSV dataset to begin."
    )