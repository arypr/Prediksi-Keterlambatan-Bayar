import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Load data
data_path = 'credit_scoring.xlsx'
df = pd.read_excel(data_path)

# Title of the app
st.title("Prediksi Keterlambatan Bayar")

# Dropdown khusus untuk pemilihan model
st.sidebar.subheader('Pilih Model yang Akan Digunakan')
models = {
    'XGBoost': 'xgboost_smote_model.pkl',
    'Adaboost': 'ada_smote_model.pkl',
    'LightGBM': 'lgbm_smote_model.pkl'
}
selected_model_name = st.sidebar.selectbox('', list(models.keys()))
model_path = models[selected_model_name]

# Load selected model
model = joblib.load(model_path)

# Pilihan antara upload dataset dan input data baru
option = st.sidebar.radio("Pilih metode input:", ("Upload Dataset", "Input Data Baru"))

if option == "Upload Dataset":
    # File uploader
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        # Determine the file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Dataset yang diunggah:")
        st.write(df.head())

        # Save original data for display
        df_original = df.copy()

        # Encode all categorical variables in df for evaluation
        df_encoded = df.copy()
        for column in df_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])

        # Drop DATE_PAY_FIRST and DATE_PAY_LAST
        df_encoded = df_encoded.drop(columns=['DATE_PAY_FIRST', 'DATE_PAY_LAST'], errors='ignore')

        X_test = df_encoded.drop(columns=['FLAG_DELINQUENT'])
        y_test = df_encoded['FLAG_DELINQUENT']

        # Check for model type and adjust feature names accordingly
        if selected_model_name == 'XGBoost':
            model_feature_names = model.get_booster().feature_names
            X_test = X_test[model_feature_names]
        elif selected_model_name == 'Adaboost':
            model_feature_names = X_test.columns.tolist()  # AdaBoost does not provide feature names directly
        elif selected_model_name == 'LightGBM':
            model_feature_names = model.feature_name()
            X_test = X_test[model_feature_names]

        # Input to select a row from X_test
        input_index = st.text_input('Masukkan Indeks Data untuk Prediksi', '')

        # Check if the input is a valid integer and within range
        try:
            selected_index = int(input_index)
            if 0 <= selected_index < len(X_test):
                # Display the selected row of original data
                input_data = df_original.iloc[selected_index]
                st.subheader('Input Data')

                # Display the data in a wider format
                st.write(input_data.to_frame().T)

                # Predict FLAG_DELINQUENT
                input_row = X_test.iloc[[selected_index]]
                predicted_flag_delinquent = model.predict(input_row)

                # Display prediction
                st.subheader('Prediksi FLAG_DELINQUENT')
                st.write(f"Predicted: {predicted_flag_delinquent[0]}")
                st.write(f"Actual: {y_test.iloc[selected_index]}")
            else:
                st.warning("Indeks tidak valid. Harap masukkan indeks yang valid dalam rentang data.")
        except ValueError:
            st.error("Input tidak valid. Harap masukkan angka.")

        # Button to show detailed evaluation
        if st.button('Show Model Evaluation and Interpretation'):
            y_pred = model.predict(X_test)
            st.subheader(f"{selected_model_name} Results:")
            st.text(classification_report(y_test, y_pred))

            # Display ROC AUC curve
            st.subheader('ROC AUC Curve')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # Display Feature Importance
            st.subheader('Feature Importance')
            if selected_model_name == 'XGBoost':
                importance = model.get_booster().get_score(importance_type='weight')
                feature_names = list(importance.keys())
                feature_importances = list(importance.values())
            elif selected_model_name == 'Adaboost':
                feature_names = X_test.columns.tolist()
                feature_importances = model.feature_importances_
            elif selected_model_name == 'LightGBM':
                importance = model.feature_importances_
                feature_names = model.feature_name()
                feature_importances = importance

            # Combine feature names and their importances into a list of tuples
            features_with_importance = list(zip(feature_importances, feature_names))

            # Sort the list of tuples by importance in descending order
            features_with_importance.sort(reverse=True, key=lambda x: x[0])

            # Unpack the sorted list into two lists: feature_importances_sorted and feature_names_sorted
            feature_importances_sorted, feature_names_sorted = zip(*features_with_importance)

            # Plot feature importance in descending order
            fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
            sns.barplot(x=feature_importances_sorted, y=feature_names_sorted, ax=ax_feat)
            ax_feat.set_title('Feature Importance')
            ax_feat.set_xlabel('Importance')
            ax_feat.set_ylabel('Feature')

            # Display the plot in Streamlit
            st.pyplot(fig_feat)

elif option == "Input Data Baru":
    st.subheader('Masukkan Data untuk Prediksi')
    selected_region = st.selectbox('Pilih Region Area', df['REGION_AREA'].unique())
    filtered_df_region = df[df['REGION_AREA'] == selected_region]

    selected_district = st.selectbox('Pilih District', filtered_df_region['DISTRICT_POS'].unique())
    filtered_df_district = filtered_df_region[filtered_df_region['DISTRICT_POS'] == selected_district]

    selected_area = st.selectbox('Pilih Area', filtered_df_district['AREA_POS'].unique())
    filtered_df_area = filtered_df_district[filtered_df_district['AREA_POS'] == selected_area]

    selected_subregion = st.selectbox('Pilih Subregion', filtered_df_area['SUBREGION_POS'].unique())
    filtered_df_subregion = filtered_df_area[filtered_df_area['SUBREGION_POS'] == selected_subregion]

    selected_business_area = st.selectbox('Pilih Business Area', filtered_df_subregion['NAME_SALES_BUSINESS_AREA'].unique())

    selected_goods_category = st.selectbox('Pilih Goods Category', df['NAME_GOODS_CATEGORY'].unique())
    selected_product_segmentation = st.selectbox('Pilih Product Segmentation', df['PRODUCT_SEGMENTATION'].unique())
    selected_owner_financing = st.selectbox('Pilih Owner Financing', df['OWNER_FINANCING'].unique())
    selected_family_status = st.selectbox('Pilih Family Status', df['NAME_FAMILY_STATUS'].unique())
    selected_client_type = st.selectbox('Pilih Client Type', df['CLIENT_TYPE'].unique())

    # Input fields for new numerical variables
    age = st.number_input('Age', min_value=0, value=0)
    days_since_first_payment = st.number_input('Days Since First Payment', min_value=0, value=0)
    amt_outstanding_principal = st.number_input('Amount Outstanding Principal', min_value=0.0, value=0.0)
    amt_instalment = st.number_input('Amount Instalment', min_value=0.0, value=0.0)
    tenure = st.number_input('Tenure', min_value=0, value=0)

    # Make predictions based on the user input
    if st.button('Predict'):
        if tenure == 0 or amt_instalment == 0:
            st.warning("Harap isi semua input numerik.")
        else:
            # Create input data for prediction
            input_data = {
                'REGION_AREA': selected_region,
                'DISTRICT_POS': selected_district,
                'AREA_POS': selected_area,
                'SUBREGION_POS': selected_subregion,
                'NAME_SALES_BUSINESS_AREA': selected_business_area,
                'NAME_GOODS_CATEGORY': selected_goods_category,
                'PRODUCT_SEGMENTATION': selected_product_segmentation,
                'OWNER_FINANCING': selected_owner_financing,
                'NAME_FAMILY_STATUS': selected_family_status,
                'CLIENT_TYPE': selected_client_type,
                'AGE': age,
                'DAYS_SINCE_FIRST_PAYMENT': days_since_first_payment,
                'AMT_OUTSTANDING_PRINCIPAL': amt_outstanding_principal,
                'AMT_INSTALMENT': amt_instalment,
                'TENURE': tenure
            }

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            for column in input_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                le.fit(df[column].astype(str))  # Fit encoder on the original dataset
                input_df[column] = le.transform(input_df[column].astype(str))

            # Drop DATE_PAY_FIRST and DATE_PAY_LAST if they exist in the input data
            input_df = input_df.drop(columns=['DATE_PAY_FIRST', 'DATE_PAY_LAST'], errors='ignore')

            if selected_model_name == 'XGBoost':
                input_df = input_df[model.get_booster().feature_names]
            elif selected_model_name == 'LightGBM':
                input_df = input_df[model.feature_name()]

            # Predict FLAG_DELINQUENT
            predicted_flag_delinquent = model.predict(input_df)
            predicted_proba = model.predict_proba(input_df)[:, 1]

            # Display prediction
            st.subheader('Prediksi FLAG_DELINQUENT')
            st.write(f"Predicted: {predicted_flag_delinquent[0]}")
            st.write(f"Probabilitas: {predicted_proba[0]:.4f}")

            # Display classification report and ROC AUC if available
            st.subheader(f"{selected_model_name} Results:")
            y_pred = model.predict(X_test)
            st.text(classification_report(y_test, y_pred))

            # Display ROC AUC curve
            st.subheader('ROC AUC Curve')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # Display Feature Importance
            st.subheader('Feature Importance')
            if selected_model_name == 'XGBoost':
                importance = model.get_booster().get_score(importance_type='weight')
                feature_names = list(importance.keys())
                feature_importances = list(importance.values())
            elif selected_model_name == 'Adaboost':
                feature_names = input_df.columns.tolist()
                feature_importances = model.feature_importances_
            elif selected_model_name == 'LightGBM':
                importance = model.feature_importances_
                feature_names = model.feature_name()
                feature_importances = importance

            # Combine feature names and their importances into a list of tuples
            features_with_importance = list(zip(feature_importances, feature_names))

            # Sort the list of tuples by importance in descending order
            features_with_importance.sort(reverse=True, key=lambda x: x[0])

            # Unpack the sorted list into two lists: feature_importances_sorted and feature_names_sorted
            feature_importances_sorted, feature_names_sorted = zip(*features_with_importance)

            # Plot feature importance in descending order
            fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
            sns.barplot(x=feature_importances_sorted, y=feature_names_sorted, ax=ax_feat)
            ax_feat.set_title('Feature Importance')
            ax_feat.set_xlabel('Importance')
            ax_feat.set_ylabel('Feature')

            # Display the plot in Streamlit
            st.pyplot(fig_feat)
