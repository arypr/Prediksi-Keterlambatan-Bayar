import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Load data
data_path = 'scoring1.xlsx'
df = pd.read_excel(data_path)

# Title of the app
st.title("Klasifikasi Keterlambatan Bayar")

# Dropdown khusus untuk pemilihan model
st.sidebar.subheader('Pilih Model yang Akan Digunakan')
models = {
    'XGBoost': 'model_xgb1.pkl',
    'LightGBM': 'model_lgb1.pkl'
}
selected_model_name = st.sidebar.selectbox('', list(models.keys()))
model_path = models[selected_model_name]

# Load selected model
model = joblib.load(model_path)

# Pilihan antara upload dataset dan input data baru
option = st.sidebar.radio("Pilih metode input:", ("Upload Dataset", "Input Data Baru"))

# Fitur yang akan digunakan dalam pemodelan
selected_features = ['PAYMENT_RATIO', 'AMT_OUTSTANDING_TOTAL', 'NAME_FAMILY_STATUS', 
                     'NAME_SALES_BUSINESS_AREA', 'AGE', 'REGION_AREA', 'CLIENT_TYPE']

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

        # Hanya gunakan fitur yang dipilih
        X_test = df_encoded[selected_features]
        y_test = df_encoded['FLAG_DELINQUENT']

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
    
    # Masukkan input data sesuai fitur yang dipilih
    selected_region = st.selectbox('Pilih Region Area', df['REGION_AREA'].unique())
    payment_ratio = df[df['REGION_AREA'] == selected_region]['PAYMENT_RATIO'].mean()
    st.write(f"PAYMENT_RATIO untuk {selected_region} adalah: {payment_ratio:.2f}")
    selected_business_area = st.selectbox('Pilih Business Area', df['NAME_SALES_BUSINESS_AREA'].unique())
    selected_family_status = st.selectbox('Pilih Family Status', df['NAME_FAMILY_STATUS'].unique())
    selected_client_type = st.selectbox('Pilih Client Type', df['CLIENT_TYPE'].unique())
    
    amt_outstanding_total = st.number_input('Amount Outstanding Total', min_value=0.0, value=0.0)
    age = st.number_input('Age', min_value=0, value=0)

    # Button to predict and show evaluation
    if st.button('Predict and Show Interpretation'):
        # Check for missing inputs
        if payment_ratio == 0.0 or amt_outstanding_total == 0.0 or age == 0:
            st.error("Harap isi semua data dengan benar.")
        else:
            # Prepare input data for prediction
            input_data = {
                'PAYMENT_RATIO': payment_ratio,
                'AMT_OUTSTANDING_TOTAL': amt_outstanding_total,
                'NAME_FAMILY_STATUS': selected_family_status,
                'NAME_SALES_BUSINESS_AREA': selected_business_area,
                'AGE': age,
                'REGION_AREA': selected_region,
                'CLIENT_TYPE': selected_client_type
            }
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            label_encoders = {}
            categorical_columns = input_df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                input_df[column] = le.fit_transform(input_df[column])

            # Tentukan nama fitur sesuai model yang dipilih
            if selected_model_name == 'XGBoost':
                model_feature_names = model.get_booster().feature_names
            elif selected_model_name == 'LightGBM':
                model_feature_names = model.booster_.feature_name()

            # Pilih kolom fitur yang relevan dari input_df
            input_df = input_df[selected_features]

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

            # Feature Importance
            st.subheader('Feature Importance')
            
            # Mendapatkan importance dari fitur model yang digunakan (misalnya XGBoost atau LightGBM)
            if selected_model_name == 'XGBoost':
                importance = model.get_booster().get_score(importance_type='weight')
                feature_names = list(importance.keys())
                feature_importances = list(importance.values())
            elif selected_model_name == 'LightGBM':
                feature_importances = model.feature_importances_
                feature_names = model.booster_.feature_name()
            
            # Membuat DataFrame untuk menampilkan feature importance
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            })
            
            # Mengurutkan fitur berdasarkan pentingnya
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            # Menampilkan feature importance dalam bentuk tabel
            st.dataframe(feature_importance_df)
            
            # Visualisasi feature importance dengan bar chart
            fig_feat, ax_feat = plt.subplots()
            ax_feat.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], align='center')
            ax_feat.set_xlabel('Feature Importance')
            ax_feat.set_title('Feature Importance')
            st.pyplot(fig_feat)

