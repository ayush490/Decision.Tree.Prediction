import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import io
import warnings

warnings.filterwarnings("ignore")

st.title("Decision Tree Classifier")

# --- Utility: Remove Outliers using IQR ---
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("Dataset Overview")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("**Description:**")
    st.write(df.describe(include='all'))

    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    st.subheader("Drop Columns (optional)")
    drop_cols = st.multiselect("Select columns to drop (optional):", df.columns)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        st.success(f"Dropped columns: {', '.join(drop_cols)}")

    st.subheader("Null Value Check")
    st.write(df.isnull().sum())

    st.subheader("Select Target Column")
    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)

    st.subheader("Boxplots for Selected Features")
    # Annotate column types and prepare options
    boxplot_options = []
    boxplot_map = {}

    for col in df.columns:
        if col == target_col:
            continue  # Skip target column
        col_type = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        label = f"{col} ({col_type})"
        boxplot_options.append(label)
        boxplot_map[label] = col

    selected_boxplot_labels = st.multiselect("Select columns for boxplots:", boxplot_options)
    selected_boxplot_cols = [boxplot_map[label] for label in selected_boxplot_labels]

    if selected_boxplot_cols:
        n = len(selected_boxplot_cols)
        fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(15, 4 * ((n + 2) // 3)))
        axes = axes.flatten()

        for i, col in enumerate(selected_boxplot_cols):
            sns.boxplot(y=df[col], ax=axes[i], color='lightblue')
            axes[i].set_title(f"Boxplot: {col}")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No columns selected for boxplot.")

    # --- Outlier Removal ---
    st.subheader("Remove Outliers (IQR Method)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = st.multiselect("Select numerical columns to remove outliers from:", numeric_cols)

    if outlier_cols:
        original_shape = df.shape
        for col in outlier_cols:
            df = remove_outliers_iqr(df, col)
        st.success(f"Removed outliers. Rows reduced from {original_shape[0]} to {df.shape[0]}.")

    # --- Label Encoding ---
    encode_labels = st.checkbox("Encode categorical labels?", value=True)
    label_encoders = {}

    if encode_labels:
        st.subheader("Label Encoding")
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        st.success("Label encoding applied.")

    # Prepare features/target after encoding
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # --- Model Training ---
    test_size = st.number_input(f"Enter train test split size", value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.subheader("Decision Tree Model Training")
    max_depth = st.slider("Select Max Depth (Pre-Pruning)", min_value=1, max_value=20, value=5)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy Score:** {acc:.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧠 Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 8))
    class_names = (
        label_encoders[target_col].classes_
        if encode_labels and target_col in label_encoders
        else [str(c) for c in y.unique()]
    )

    fontsize = st.slider("Font size", min_value=1, max_value=20, value=8)
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax
    )
    st.pyplot(fig)

    # --- Prediction ---
    st.subheader("🧪 Predict on New Input")
    user_input = {}
    for col in X.columns:
        if encode_labels and col in label_encoders:
            options = label_encoders[col].classes_
            user_input[col] = st.selectbox(f"Select {col}", options)
        else:
            user_input[col] = st.number_input(f"Enter numeric value for {col}", step=1.0)

    if st.button("Predict"):
        input_row = pd.DataFrame([{
            col: label_encoders[col].transform([user_input[col]])[0]
            if encode_labels and col in label_encoders else user_input[col]
            for col in X.columns
        }])
        prediction = model.predict(input_row)
        predicted_class = (
            label_encoders[target_col].inverse_transform(prediction)[0]
            if encode_labels and target_col in label_encoders
            else prediction[0]
        )
        st.success(f"Predicted Class: {predicted_class}")
