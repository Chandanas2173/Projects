import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv(r'C:\Users\wwwsa\Downloads\survey lung cancer.csv')

# Apply Label Encoding
data['LUNG_CANCER'] = LabelEncoder().fit_transform(data['LUNG_CANCER'])
data['GENDER'] = LabelEncoder().fit_transform(data['GENDER'])

# Prepare Data for Model Training
X = data[['YELLOW_FINGERS', 'ALLERGY ', 'WHEEZING', 'COUGHING', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']]
y = data['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model Once
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute Model Performance Once
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
classification_rep = classification_report(y_test, y_pred)


def main():
    st.title("Lung Cancer Prediction App")

    tab1, tab2 = st.tabs(["Exploratory Data Analysis (EDA)", "Prediction"])

    with st.sidebar:
        st.header("Enter Your Symptoms")
        feature_questions = {
            'YELLOW_FINGERS': "Do you have yellow fingers?",
            'ALLERGY ': "Do you have allergies?",
            'WHEEZING': "Do you experience wheezing?",
            'COUGHING': "Do you have coughing?",
            'SWALLOWING DIFFICULTY': "Do you have difficulty swallowing?",
            'CHEST PAIN': "Do you have chest pain?"
        }

        user_input = {}
        for feature, question in feature_questions.items():
            user_input[feature] = st.selectbox(question, ["No", "Yes"], index=0)

        # Prediction and Reset Buttons
        col1, col2 = st.columns(2)

        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = ""

        with col1:
            if st.button("Predict"):
                user_df = pd.DataFrame([{k: 1 if v == "Yes" else 0 for k, v in user_input.items()}])
                prediction = model.predict(user_df)
                st.session_state.prediction_result = "Yes, you have Lung Cancer" if prediction[
                                                                                        0] == 1 else "No, you don't have Lung Cancer"
                st.rerun()

        with col2:
            if st.button("Reset"):
                st.session_state.prediction_result = ""
                st.rerun()

    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        st.write("### Data Overview")
        st.write(data.head())

        st.write("### Data Distribution")
        st.write(data.describe())

        st.write("### Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='LUNG_CANCER', ax=ax)
        ax.set_xticklabels(["No", "Yes"])
        st.pyplot(fig)

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8}, ax=ax)
        st.pyplot(fig)

        st.write("### Feature Importance")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        feature_importances.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.write("### Feature Distributions")
        feature_columns = ['YELLOW_FINGERS', 'ALLERGY ', 'WHEEZING', 'COUGHING', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, feature in enumerate(feature_columns):
            sns.countplot(data=data, x=feature, ax=axes[i])
            axes[i].set_title(feature)
            axes[i].set_xticklabels(["No", "Yes"])

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.header("Lung Cancer Prediction")

        # Display Prediction Result
        if st.session_state.prediction_result:
            st.write(f"### Lung Cancer Prediction: {st.session_state.prediction_result}")

        # Model Performance
        st.write("### Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}%")
        st.text("Classification Report:")
        st.text(classification_rep)


if __name__ == "__main__":
    main()
