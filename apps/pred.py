#Packages related to general operating system & warnings
import os 
import warnings
warnings.filterwarnings('ignore')

# importing required libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


def app():
    st.title("Prediction")
    image=Image.open("banking_app.png")
    st.image(image,use_column_width=True)
    hide_menu_style = """
            <style>
            #MainMenu {visibility:hidden; }
            footer {visibility:hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    def user_input_features():
        col1,col2,col3 = st.columns((2,1,1))
        age = col1.slider('Age',17,98,40)

        job = col2.selectbox('Job',('housemaid','services','admin.','blue-collar'
                                ,'technician','retired','management','unemployed'
                                ,'self-employed','entrepreneur','student'))

        marital = col3.selectbox('Marital Status',('married','single','divorced'))    

        col4,col5,col6,col7 = st.columns((1,1,1,1))  

        education = col4.selectbox('Education',('basic.4y','high.school','basic.6y','basic.9y'
                                            ,'professional.course','university.degree','illiterate'))                  
        
        default = col5.selectbox('Credit Defaults',('no','yes'))
        housing = col6.selectbox('Housing Loan',('no','yes'))
        loan = col7.selectbox('Other Loan',('no','yes'))

        col8,col9,col10 = st.columns((2,1,1))

        previous = col8.slider('Previously Contacted',0,7,0) 
        emp_var_rate = col9.selectbox('Employment variation Rate (Quarterly)',(1.1,  1.4, -0.1, -0.2, -1.8, -2.9, -3.4, -3. , -1.7, -1.1))
        cons_price_idx= col10.selectbox('Consumer Price Index (Monthly)',(93.994, 94.465, 93.918, 93.444, 93.798, 93.2  , 92.756, 92.843,93.075, 92.893, 
                                                                        92.963, 92.469, 92.201, 92.379, 92.431, 92.649,92.713, 93.369, 93.749, 93.876, 
                                                                        94.055, 94.215, 94.027, 94.199,94.601, 94.767)) 

        col11,col12,col13 = st.columns((1,1,1))

        cons_conf_idx = col11.selectbox('Consumer Confidence Index (Monthly)',(-36.4, -41.8, -42.7, -36.1, -40.4, -42. , -45.9, -50. , -47.1,-46.2, -40.8,
                                                                         -33.6, -31.4, -29.8, -26.9, -30.1, -33. , -34.8,-34.6, -40. , -39.8, -40.3, -38.3,
                                                                          -37.5, -49.5, -50.8))
    
        euribor3m = col11.selectbox('Euribor 3 Months Rate (Daily)',(4.857, 4.856, 4.855, 4.859, 4.86 , 4.858, 4.864, 4.865, 4.866,4.967, 4.961, 4.959, 4.958, 
                                                                    4.96 , 4.962, 4.955, 4.947, 4.956,4.966, 4.963, 4.957, 4.968, 4.97 , 4.965, 4.964, 5.045, 5.   ,
                                                                    4.936, 4.921, 4.918, 4.912, 4.827, 4.794, 4.76 , 4.733, 4.7  ,
                                                                    4.663, 4.592, 4.474, 4.406, 4.343, 4.286, 4.245, 4.223, 4.191,
                                                                    4.153, 4.12 , 4.076, 4.021, 3.901, 3.879, 3.853, 3.816, 3.743,
                                                                    3.669, 3.563, 3.488, 3.428, 3.329, 3.282, 3.053, 1.811, 1.799,
                                                                    1.778, 1.757, 1.726, 1.703, 1.687, 1.663, 1.65 , 1.64 , 1.629,
                                                                    1.614, 1.602, 1.584, 1.574, 1.56 , 1.556, 1.548, 1.538, 1.531,
                                                                    1.52 , 1.51 , 1.498, 1.483, 1.479, 1.466, 1.453, 1.445, 1.435,
                                                                    1.423, 1.415, 1.41 , 1.405, 1.406, 1.4  , 1.392, 1.384, 1.372,
                                                                    1.365, 1.354, 1.344, 1.334, 1.327, 1.313, 1.299, 1.291, 1.281,
                                                                    1.266, 1.25 , 1.244, 1.259, 1.264, 1.27 , 1.262, 1.26 , 1.268,
                                                                    1.286, 1.252, 1.235, 1.224, 1.215, 1.206, 1.099, 1.085, 1.072,
                                                                    1.059, 1.048, 1.044, 1.029, 1.018, 1.007, 0.996, 0.979, 0.969,
                                                                    0.944, 0.937, 0.933, 0.927, 0.921, 0.914, 0.908, 0.903, 0.899,
                                                                    0.884, 0.883, 0.881, 0.879, 0.873, 0.869, 0.861, 0.859, 0.854,
                                                                    0.851, 0.849, 0.843, 0.838, 0.834, 0.829, 0.825, 0.821, 0.819,
                                                                    0.813, 0.809, 0.803, 0.797, 0.788, 0.781, 0.778, 0.773, 0.771,
                                                                    0.77 , 0.768, 0.766, 0.762, 0.755, 0.749, 0.743, 0.741, 0.739,
                                                                    0.75 , 0.753, 0.754, 0.752, 0.744, 0.74 , 0.742, 0.737, 0.735,
                                                                    0.733, 0.73 , 0.731, 0.728, 0.724, 0.722, 0.72 , 0.719, 0.716,
                                                                    0.715, 0.714, 0.718, 0.721, 0.717, 0.712, 0.71 , 0.709, 0.708,
                                                                    0.706, 0.707, 0.7  , 0.655, 0.654, 0.653, 0.652, 0.651, 0.65 ,
                                                                    0.649, 0.646, 0.644, 0.643, 0.639, 0.637, 0.635, 0.636, 0.634,
                                                                    0.638, 0.64 , 0.642, 0.645, 0.659, 0.663, 0.668, 0.672, 0.677,
                                                                    0.682, 0.683, 0.684, 0.685, 0.688, 0.69 , 0.692, 0.695, 0.697,
                                                                    0.699, 0.701, 0.702, 0.704, 0.711, 0.713, 0.723, 0.727, 0.729,
                                                                    0.732, 0.748, 0.761, 0.767, 0.782, 0.79 , 0.793, 0.802, 0.81 ,
                                                                    0.822, 0.827, 0.835, 0.84 , 0.846, 0.87 , 0.876, 0.885, 0.889,
                                                                    0.893, 0.896, 0.898, 0.9  , 0.904, 0.905, 0.895, 0.894, 0.891,
                                                                    0.89 , 0.888, 0.886, 0.882, 0.88 , 0.878, 0.877, 0.942, 0.953,
                                                                    0.956, 0.959, 0.965, 0.972, 0.977, 0.982, 0.985, 0.987, 0.993,
                                                                    1.   , 1.008, 1.016, 1.025, 1.032, 1.037, 1.043, 1.045, 1.047,
                                                                    1.05 , 1.049, 1.046, 1.041, 1.04 , 1.039, 1.035, 1.03 , 1.031,
                                                                    1.028))
        
         

        nr_employed = col12.selectbox('Number of Employed Citizens (Quarterly)',( 5191, 5228, 5195, 5176, 5099, 5076, 5017, 5023, 5008, 4991, 4963))

        data = {'age':age,
                'job':job,
                'marital':marital,
                'education':education,
                'default':default,
                'housing':housing,
                'loan':loan,
                'previous':previous,
                'emp_var_rate':emp_var_rate,
                'cons_price_idx':cons_price_idx,
                'cons_conf_idx':cons_conf_idx,
                'euribor3m':euribor3m,
                'nr_employed':nr_employed}

        features = pd.DataFrame(data,index=[0])
        features['emp_var_rate'] = features['emp_var_rate'].astype('int64')
        features['cons_price_idx'] = features['cons_price_idx'].astype('int64')
        features['cons_conf_idx'] = features['cons_conf_idx'].astype('int64')
        features['euribor3m'] = features['euribor3m'].astype('int64')
        features['nr_employed'] = features['nr_employed'].astype('int64')    

        return features


    input_df = user_input_features()

    data_raw = pd.read_csv('bank-additional-full.csv',sep=';')
    data_raw.rename(columns={"emp.var.rate":"emp_var_rate","cons.price.idx":"cons_price_idx","cons.conf.idx":"cons_conf_idx","nr.employed":"nr_employed"},inplace=True)
    data_raw['nr_employed']=data_raw['nr_employed'].astype('int64')
    data_raw['target'] = data_raw.apply(lambda row: 1 if row["y"] == "yes" else 0, axis=1)
    data_raw = data_raw.drop_duplicates(keep='last')
    data_raw.replace(to_replace="unknown", value=np.nan, inplace=True)
    data_raw = data_raw.apply(lambda x: x.fillna(x.value_counts().index[0]))

    #handling the imbalance dataset
    data1=data_raw.copy()
    data2=data1[data1.target==1]
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    bal_data=data1
    
    bal_data = bal_data[['age','job','marital','education','default','housing','loan'
                ,'previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']] 

    data = pd.concat([input_df,bal_data],axis=0)

    #Label encoding
    le = preprocessing.LabelEncoder()
    data.job = le.fit_transform(data.job)
    data.marital = le.fit_transform(data.marital)
    data.education = le.fit_transform(data.education)
    data.default = le.fit_transform(data.default)
    data.housing = le.fit_transform(data.housing)
    data.loan = le.fit_transform(data.loan)

    data = data[:1] 

    load_clf = pickle.load(open('bank_clf.pkl','rb'))
    

    prediction = load_clf.predict(data)  
    prediction_proba = load_clf.predict_proba(data) 

    col14,col15 = st.columns(2)
    col14.subheader('Prediction')
    subscription = np.array(['no','yes'])
    col14.write(subscription[prediction])  

    col15.subheader('Prediction Probability') 
    col15.write(prediction_proba)     
