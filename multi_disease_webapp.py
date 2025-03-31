import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai



#load saved model

diabetes_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/diabetes_model.sav",'rb'))
heart_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/heart_model.sav",'rb'))
parkinsons_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/parkinsons_model.sav",'rb'))
kidney_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/kidney_disease_model.sav",'rb'))
breast_cancer_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/breast_canser_model.sav",'rb'))
liver_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/liver_patient_model.sav",'rb'))
lung_cancer_model=pickle.load(open("C:/Users/mudgo/OneDrive/Desktop/ml prj/models/lung_cancer_model.sav",'rb'))


#side bar
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System using ML",
        [
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Parkinsons Prediction',
            'Kidney Disease Prediction',
            'Breast Cancer Prediction',
            'Liver Disease Prediction',
            'Lung Cancer Prediction',
            'AI Health Chatbot'
        ],
        icons=[
            'activity', 
            'heart-pulse',  
            'person-wheelchair',  
            'droplet', 
            'gender-female',
            'clipboard-pulse',
            'lungs',
            'chat-dots'
        ],
        default_index=-1
    )
    

#Pages
#Diabetes
if selected=='Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    col1,col2,col3=st.columns(3)

    with col1:
        Pregnancies=st.text_input('Number of pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    #code for prediction
    diab_diagnosis=''

    #button for predction
    # Code for prediction


    # Button for prediction
    if st.button("Diabetes Test Result"):
        try:
            # Convert inputs to float
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                        float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]

            # Make prediction
            diab_prediction = diabetes_model.predict([user_input])
            
            # Display result using Streamlit
            if diab_prediction[0] == 1:
                diab_diagnosis = "The person is Diabetic"
                st.error(diab_diagnosis)  # Display in red (error)
            else:
                diab_diagnosis = "The person is not Diabetic"
                st.success(diab_diagnosis)  # Display in green (success)
        
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")



#Heart

if selected=='Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        # sex = st.text_input('Sex (1 = Male, 0 = Female)')
        sex=st.selectbox('sex',['Male','Female'])
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar (> 120 mg/dl, 1 = True, 0 = False)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')
    with col3:
        ca = st.text_input('Number of Major Vessels (0-3)')
    with col1:
        thal = st.text_input('Thalassemia (0-3)')

    gender=1 if sex=='Male' else 0
    # Code for prediction
    heart_diagnosis = ''

    # Button for prediction
    if st.button("Heart Disease Test Result"):
        try:
            heart_prediction = heart_model.predict([[float(age), gender, float(cp), float(trestbps),
                                                float(chol), float(fbs), float(restecg), float(thalach),
                                                float(exang), float(oldpeak), float(slope),
                                                float(ca), float(thal)]])
            
            if heart_prediction[0] == 1:
                heart_diagnosis = "The person has Heart Disease"
                st.error(heart_diagnosis)
            else:
                heart_diagnosis = "The person does not have Heart Disease"
                st.success(heart_diagnosis)

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")
        

#parkinsons
if selected=='Parkinsons Prediction':
    st.title('Parkinsons Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo_Hz = st.text_input('MDVP:Fo(Hz) - Average Vocal Frequency')
    with col2:
        MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz) - Maximum Vocal Frequency')
    with col3:
        MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz) - Minimum Vocal Frequency')
    with col1:
        MDVP_Jitter = st.text_input('MDVP:Jitter(%)')
    with col2:
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
        MDVP_RAP = st.text_input('MDVP:RAP')
    with col1:
        MDVP_PPQ = st.text_input('MDVP:PPQ')
    with col2:
        Jitter_DDP = st.text_input('Jitter:DDP')
    with col3:
        MDVP_Shimmer = st.text_input('MDVP:Shimmer')
    with col1:
        MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col2:
        Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
    with col3:
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
    with col1:
        MDVP_APQ = st.text_input('MDVP:APQ')
    with col2:
        Shimmer_DDA = st.text_input('Shimmer:DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col1:
        spread1 = st.text_input('Spread1')
    with col2:
        spread2 = st.text_input('Spread2')
    with col3:
        D2 = st.text_input('D2')
    with col1:
        PPE = st.text_input('PPE')

    # Code for prediction
    parkinsons_diagnosis = ''

    # Button for prediction
    if st.button("Parkinson’s Disease Test Result"):
        try:
            parkinsons_prediction = parkinsons_model.predict([[float(MDVP_Fo_Hz), float(MDVP_Fhi_Hz), float(MDVP_Flo_Hz),
                                                                float(MDVP_Jitter), float(MDVP_Jitter_Abs), float(MDVP_RAP),
                                                                float(MDVP_PPQ), float(Jitter_DDP), float(MDVP_Shimmer),
                                                                float(MDVP_Shimmer_dB), float(Shimmer_APQ3), float(Shimmer_APQ5),
                                                                float(MDVP_APQ), float(Shimmer_DDA), float(NHR), float(HNR),
                                                                float(RPDE), float(DFA), float(spread1), float(spread2),
                                                                float(D2), float(PPE)]])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson’s Disease"
                st.error(parkinsons_diagnosis)
            else:
                parkinsons_diagnosis = "The person does not have Parkinson’s Disease"
                st.success(parkinsons_diagnosis)
        
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")

    

#kidney
if selected=='Kidney Disease Prediction':
    st.title('Chronic Kidney Disease Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Bp = st.text_input('Blood Pressure (mm Hg)')
    with col2:
        Sg = st.text_input('Specific Gravity')
    with col3:
        Al = st.text_input('Albumin Level')
    with col1:
        Su = st.text_input('Sugar Level')
    with col2:
        Rbc = st.text_input('Red Blood Cells (0 = Normal, 1 = Abnormal)')
    with col3:
        Bu = st.text_input('Blood Urea (mg/dL)')
    with col1:
        Sc = st.text_input('Serum Creatinine (mg/dL)')
    with col2:
        Sod = st.text_input('Sodium Level (mEq/L)')
    with col3:
        Pot = st.text_input('Potassium Level (mEq/L)')
    with col1:
        Hemo = st.text_input('Hemoglobin Level (g/dL)')
    with col2:
        Wbcc = st.text_input('White Blood Cell Count (cells/cumm)')
    with col3:
        Rbcc = st.text_input('Red Blood Cell Count (millions/cumm)')
    with col1:
        Htn = st.text_input('Hypertension (0 = No, 1 = Yes)')

    # Code for prediction
    ckd_diagnosis = ''

    # Button for prediction
    if st.button("CKD Test Result"):
        try:
            ckd_prediction = kidney_model.predict([[float(Bp), float(Sg), float(Al), float(Su), float(Rbc),
                                                float(Bu), float(Sc), float(Sod), float(Pot), float(Hemo),
                                                float(Wbcc), float(Rbcc), float(Htn)]])

            if ckd_prediction[0] == 1:
                ckd_diagnosis = "The person has Chronic Kidney Disease"
                st.error(ckd_diagnosis)
            else:
                ckd_diagnosis = "The person does not have Chronic Kidney Disease"
                st.success(ckd_diagnosis)

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")
        


#breast cancer
if selected=='Breast Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.text_input('Mean Radius')
    with col2:
        texture_mean = st.text_input('Mean Texture')
    with col3:
        perimeter_mean = st.text_input('Mean Perimeter')
    with col1:
        area_mean = st.text_input('Mean Area')
    with col2:
        smoothness_mean = st.text_input('Mean Smoothness')
    with col3:
        compactness_mean = st.text_input('Mean Compactness')
    with col1:
        concavity_mean = st.text_input('Mean Concavity')
    with col2:
        concave_points_mean = st.text_input('Mean Concave Points')
    with col3:
        symmetry_mean = st.text_input('Mean Symmetry')
    with col1:
        fractal_dimension_mean = st.text_input('Mean Fractal Dimension')
    with col2:
        radius_se = st.text_input('Radius SE')
    with col3:
        texture_se = st.text_input('Texture SE')
    with col1:
        perimeter_se = st.text_input('Perimeter SE')
    with col2:
        area_se = st.text_input('Area SE')
    with col3:
        smoothness_se = st.text_input('Smoothness SE')
    with col1:
        compactness_se = st.text_input('Compactness SE')
    with col2:
        concavity_se = st.text_input('Concavity SE')
    with col3:
        concave_points_se = st.text_input('Concave Points SE')
    with col1:
        symmetry_se = st.text_input('Symmetry SE')
    with col2:
        fractal_dimension_se = st.text_input('Fractal Dimension SE')
    with col3:
        radius_worst = st.text_input('Worst Radius')
    with col1:
        texture_worst = st.text_input('Worst Texture')
    with col2:
        perimeter_worst = st.text_input('Worst Perimeter')
    with col3:
        area_worst = st.text_input('Worst Area')
    with col1:
        smoothness_worst = st.text_input('Worst Smoothness')
    with col2:
        compactness_worst = st.text_input('Worst Compactness')
    with col3:
        concavity_worst = st.text_input('Worst Concavity')
    with col1:
        concave_points_worst = st.text_input('Worst Concave Points')
    with col2:
        symmetry_worst = st.text_input('Worst Symmetry')
    with col3:
        fractal_dimension_worst = st.text_input('Worst Fractal Dimension')

    # Code for prediction
    cancer_diagnosis = ''

    # Button for prediction
    if st.button("Breast Cancer Test Result"):
        try:
            cancer_prediction = breast_cancer_model.predict([[float(radius_mean), float(texture_mean), float(perimeter_mean),
                                                            float(area_mean), float(smoothness_mean), float(compactness_mean),
                                                            float(concavity_mean), float(concave_points_mean), float(symmetry_mean),
                                                            float(fractal_dimension_mean), float(radius_se), float(texture_se),
                                                            float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se),
                                                            float(concavity_se), float(concave_points_se), float(symmetry_se),
                                                            float(fractal_dimension_se), float(radius_worst), float(texture_worst),
                                                            float(perimeter_worst), float(area_worst), float(smoothness_worst),
                                                            float(compactness_worst), float(concavity_worst), float(concave_points_worst),
                                                            float(symmetry_worst), float(fractal_dimension_worst)]])

            if cancer_prediction[0] == 1:
                cancer_diagnosis = "The person has Breast Cancer (Malignant)"
                st.error(cancer_diagnosis)
            else:
                cancer_diagnosis = "The person does not have Breast Cancer (Benign)"
                st.success(cancer_diagnosis)

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")

    

#Liver
if selected=='Liver Disease Prediction':
    st.title('Liver Disease Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age of the Person')
    with col2:
        Gender = st.selectbox('Gender', ['Male', 'Female'])
    with col3:
        Total_Bilirubin = st.text_input('Total Bilirubin')
    with col1:
        Direct_Bilirubin = st.text_input('Direct Bilirubin')
    with col2:
        Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase Level')
    with col3:
        Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase Level')
    with col1:
        Aspartate_Aminotransferase = st.text_input('Aspartate Aminotransferase Level')
    with col2:
        Total_Protiens = st.text_input('Total Proteins Level')
    with col3:
        Albumin = st.text_input('Albumin Level')
    with col1:
        Albumin_and_Globulin_Ratio = st.text_input('Albumin and Globulin Ratio')

    # Convert categorical Gender to binary (0 for Female, 1 for Male)
    gender_binary = 1 if Gender == 'Male' else 0

    # Code for prediction
    liver_diagnosis = ''

    # Button for prediction
    if st.button("Liver Disease Test Result"):
        try:
            liver_prediction = liver_model.predict([[float(Age), gender_binary, float(Total_Bilirubin),
                                                            float(Direct_Bilirubin), float(Alkaline_Phosphotase),
                                                            float(Alamine_Aminotransferase), float(Aspartate_Aminotransferase),
                                                            float(Total_Protiens), float(Albumin), float(Albumin_and_Globulin_Ratio)]])

            if liver_prediction[0] == 1:
                liver_diagnosis = "The person is likely to have Liver Disease."
                st.error(liver_diagnosis)
            else:
                liver_diagnosis = "The person is unlikely to have Liver Disease."
                st.success(liver_diagnosis)
        
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")
        
    

#Lung
if selected=='Lung Cancer Prediction':
    st.title('Lung Cancer Prediction using ML')

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender', ['Male', 'Female'])
    with col2:
        Age = st.text_input('Age of the Person')
    with col3:
        Smoking = st.text_input('Smoking (0/1/2)')
    with col1:
        Yellow_Fingers = st.text_input('Yellow Fingers (0/1/2)')
    with col2:
        Anxiety = st.text_input('Anxiety (0/1/2)')
    with col3:
        Peer_Pressure = st.text_input('Peer Pressure (0/1/2)')
    with col1:
        Chronic_Disease = st.text_input('Chronic Disease (0/1/2)')
    with col2:
        Fatigue = st.text_input('Fatigue (0/1/2)')
    with col3:
        Allergy = st.text_input('Allergy (0/1/2)')
    with col1:
        Wheezing = st.text_input('Wheezing (0/1/2)')
    with col2:
        Alcohol_Consuming = st.text_input('Alcohol Consuming (0/1/2)')
    with col3:
        Coughing = st.text_input('Coughing (0/1/2)')
    with col1:
        Shortness_of_Breath = st.text_input('Shortness of Breath (0/1/2)')
    with col2:
        Swallowing_Difficulty = st.text_input('Swallowing Difficulty (0/1/2)')
    with col3:
        Chest_Pain = st.text_input('Chest Pain (0/1/2)')

    # Convert categorical Gender to binary (0 for Female, 1 for Male)
    gender_binary = 1 if Gender == 'Male' else 0

    # Code for prediction
    lung_cancer_diagnosis = ''

    # Button for prediction
    if st.button("Lung Cancer Test Result"):
        try:
            lung_cancer_prediction = lung_cancer_model.predict([[gender_binary, float(Age), float(Smoking),
                                                                float(Yellow_Fingers), float(Anxiety), float(Peer_Pressure),
                                                                float(Chronic_Disease), float(Fatigue), float(Allergy),
                                                                float(Wheezing), float(Alcohol_Consuming), float(Coughing),
                                                                float(Shortness_of_Breath), float(Swallowing_Difficulty),
                                                                float(Chest_Pain)]])

            if lung_cancer_prediction[0] == 1:
                lung_cancer_diagnosis = "The person is likely to have Lung Cancer."
                st.error(lung_cancer_diagnosis)
            else:
                lung_cancer_diagnosis = "The person is unlikely to have Lung Cancer."
                st.success(lung_cancer_diagnosis)
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")



#chatbot
if selected=='AI Health Chatbot':

    # Configure Gemini API key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # Store in .streamlit/secrets.toml

    # Set up Streamlit UI
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("AI Health Chatbot 🤖")

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("Ask me anything about health and diseases...")

    if user_input:
        # Store and display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Use a supported Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash-latest")  # Or use "gemini-1.5-pro-latest"
        
        try:
            response = model.generate_content(user_input)
            ai_reply = response.text

            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(ai_reply)

            # Store AI response in history
            st.session_state["messages"].append({"role": "assistant", "content": ai_reply})

        except Exception as e:
            st.error(f"Error: {e}")
