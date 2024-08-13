import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
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
    'catboost Classifier': 'catboostclass_smote_model.pkl',
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
        df_encoded = df_encoded.drop(columns=['DATE_PAY_FIRST', 'DATE_PAY_LAST'])

        X_test = df_encoded.drop(columns=['FLAG_DELINQUENT'])
        y_test = df_encoded['FLAG_DELINQUENT']

        if selected_model_name == 'XGBoost':
            X_test = X_test[model.get_booster().feature_names]
        elif selected_model_name == 'catboost Classifier':
            model_feature_names = model.feature_names_
            X_test = X_test[model_feature_names]
        elif selected_model_name == 'LightGBM':
            model_feature_names = model.booster_.feature_name()  # Fixed method
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
                predicted_flag_delinquent = model.predict(X_test.iloc[[selected_index]])

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
            elif selected_model_name == 'catboost Classifier':
                feature_names = model.feature_names_
                feature_importances = model.get_feature_importance()
            elif selected_model_name == 'LightGBM':
                importance = model.feature_importances_  # Fixed method
                feature_names = model.booster_.feature_name()  # Fixed method
                feature_importances = importance

            fig_feat, ax_feat = plt.subplots()
            ax_feat.barh(feature_names, feature_importances, align='center')
            ax_feat.set_yticks(range(len(feature_names)))
            ax_feat.set_yticklabels(feature_names)
            ax_feat.set_xlabel('Feature Importance')
            ax_feat.set_title('Feature Importance')
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

    # Button to predict and show evaluation
    if st.button('Predict and Show Interpretation'):
        # Check for missing inputs
        if tenure == 0 or amt_instalment == 0.0 or amt_outstanding_principal == 0.0:
            st.error("Harap isi semua data dengan benar.")
        else:
            # Prepare input data for prediction
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
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            label_encoders = {}
            categorical_columns = input_df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                input_df[column] = le.fit_transform(input_df[column])

            # Drop DATE_PAY_FIRST and DATE_PAY_LAST from df
            input_df = input_df.drop(columns=['DATE_PAY_FIRST', 'DATE_PAY_LAST'], errors='ignore')

            # Tentukan nama fitur sesuai model yang dipilih
            if selected_model_name == 'XGBoost':
                model_feature_names = model.get_booster().feature_names
            elif selected_model_name == 'catboost Classifier':
                model_feature_names = model.feature_names_
            elif selected_model_name == 'LightGBM':
                model_feature_names = model.booster_.feature_name()

            # Pilih kolom fitur yang relevan dari input_df
            input_df = input_df[model_feature_names]

            # Reshape the input data for prediction
            input_data_reshaped = input_df.values.reshape(1, -1)

            # Predict FLAG_DELINQUENT
            predicted_flag_delinquent = model.predict(input_data_reshaped)
            predicted_probabilities = model.predict_proba(input_data_reshaped)

            # Display prediction
            st.subheader('Prediksi FLAG_DELINQUENT')

            # Determine prediction
            prediction_label = 'Terjadi Keterlambatan' if predicted_flag_delinquent[0] == 1 else 'Tidak Terjadi Keterlambatan'
            prediction_color = '#F44336' if predicted_flag_delinquent[0] == 1 else '#4CAF50'

            # Display prediction with a colored indicator
            st.markdown(f"**Prediksi:** <span style='color:{prediction_color};'>{prediction_label}</span>", unsafe_allow_html=True)

            # Display probabilities
            prob_labels = ['Tidak Terjadi Keterlambatan', 'Terjadi Keterlambatan']
            prob_values = [predicted_probabilities[0][0], predicted_probabilities[0][1]]

            # Display probabilities with the word "Probabilitas"
            for label, value in zip(prob_labels, prob_values):
                st.markdown(f"**Probabilitas {label}:** {value:.2f}")

            # Recommendation based on prediction
            st.subheader('Rekomendasi Tindakan')
            if predicted_flag_delinquent[0] == 1:
                st.write("Rekomendasi: Pertimbangkan untuk memberikan pengingat pembayaran kepada pelanggan.")
            else:
                st.write("Rekomendasi: Pelanggan ini diprediksi tidak akan terlambat membayar.")

            # Feature Importance for explanation
            st.subheader('Feature Importance')

            # Get feature importances from the model
            if selected_model_name == 'XGBoost':
                importance = model.get_booster().get_score(importance_type='weight')
                feature_names = list(importance.keys())
                feature_importances = list(importance.values())
            elif selected_model_name == 'catboost Classifier':
                feature_names = model.feature_names_
                feature_importances = model.get_feature_importance()
            elif selected_model_name == 'LightGBM':
                feature_names = model.booster_.feature_name()
                feature_importances = model.feature_importances_

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
