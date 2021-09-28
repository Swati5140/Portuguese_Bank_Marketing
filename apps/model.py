#Packages related to general operating system & warnings
import os 
import warnings
warnings.filterwarnings('ignore')

# importing required libraries and packages
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# Data Preprocessing
# creating a new column named "pdays2" based on the value in "pdays" column 
def function (row):
    if(row['pdays']==999):
        return 0;
    return 1;

# changing the value 999 in pdays column to  value 30 
def function1 (row):
    if(row['pdays']==999):
        return 30;
    return row['pdays'];



def app():
    hide_menu_style = """
            <style>
            #MainMenu {visibility:hidden; }
            footer {visibility:hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    # Dividing the page into 3 columns(col1=sidebar, col2 and col3 = page contents)
    col1 = st.sidebar
    col2, col3 = st.columns((2,1))
    
    #Load the Dataset
    with col1.header('Upload your CSV data'):
        uploaded_file = col1.file_uploader("Upload your input CSV file",type=["csv"])
        
    if uploaded_file is not None:
        st.info('Awaiting for CSV file to be uploaded.')
        data = pd.read_csv(uploaded_file,sep=';')
        col2.subheader('**Dataset**')

    else:
        data=pd.read_csv('bank-additional-full.csv',sep=';')
        col2.subheader('**Dataset**')
        
            
        
    data.rename(columns={"emp.var.rate":"emp_var_rate","cons.price.idx":"cons_price_idx","cons.conf.idx":"cons_conf_idx","nr.employed":"nr_employed"},inplace=True)
    data['nr_employed']=data['nr_employed'].astype('int64')
    data['target'] = data.apply(lambda row: 1 if row["y"] == "yes" else 0, axis=1)
    data = data.drop_duplicates(keep='last')

    # dropping duplicates
    duplicate_data = data[data.duplicated(keep = "last")]
    data = data.drop_duplicates(keep='last')

    # Setting up numeric (num_data) and categoric (cat_data) dataframes
    num_data = data.copy().select_dtypes(include=["float64","int64"])
    cat_data = data.copy().select_dtypes(exclude=["float64","int64"])

    #Replacing 'unknown' by NaN
    cat_data.replace(to_replace="unknown", value=np.nan, inplace=True)
    
    #removing 'duration','default','day_of_week' attributes
    cat_data = cat_data.drop(['default','day_of_week'],axis=1)
    num_data = num_data.drop('duration',axis=1)

    # Imputation of missing values by the modal value
    cat_data_imputed = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Handling Outliers
    #age
    num_data.loc[num_data['age']>=69.5,'age'] = 69.5
    #campaign
    num_data.loc[num_data['campaign']<=1,'campaign'] = 1
    num_data.loc[num_data['campaign']>=5,'campaign'] = 5

    #cons_conf_idx
    num_data.loc[num_data['cons_conf_idx']>=-36,'cons_conf_idx'] = -36
    
    #pdays
    num_data['pdays2']=num_data.apply(lambda row: function(row),axis=1)
    num_data['pdays']=num_data.apply(lambda row: function1(row),axis=1)

    #changing the type of pdays to int
    num_data['pdays']=num_data['pdays'].astype('int64')
    #renaming column pdays to pdays1
    num_data.rename(columns={'pdays': 'pdays1'},inplace=True)

    data_new = pd.concat([num_data,cat_data_imputed],axis=1)
        #new_df = st.dataframe(data_new.head(50))
    
    #handling the imbalance dataset
    data1=data_new.copy()
    data2=data1[data1.target==1]
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    data1=pd.concat([data1, data2])
    bal_data=data1.copy()

    #Label encoding
    le_data = bal_data.copy()
    le = preprocessing.LabelEncoder()
    le_data.job = le.fit_transform(le_data.job)
    le_data.marital = le.fit_transform(le_data.marital)
    le_data.education = le.fit_transform(le_data.education)
    le_data.housing = le.fit_transform(le_data.housing)
    le_data.loan = le.fit_transform(le_data.loan)
    le_data.contact = le.fit_transform(le_data.contact)
    le_data.month = le.fit_transform(le_data.month)
    le_data.poutcome = le.fit_transform(le_data.poutcome)
    le_data.y = le.fit_transform(le_data.y)

    
    col2.dataframe(le_data.head(50))

    # Dividing the label encoded dataset into independent and dependent variables
    X = le_data.iloc[:, : -1].values
    y = le_data.iloc[:, -1].values
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    with col1.header('Set Random Forest Classifier Parameters'):
        split_size = col1.slider('Data split ratio (% Training Set)',10,90,70,5)

    
    # Splitting into train & test data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=(100-split_size)/100)

    #feature scaling
    scaler = StandardScaler()
    X_train.loc[:, : 10] = scaler.fit_transform(X_train.loc[:, : 10])
    X_test.loc[:, : 10] = scaler.transform(X_test.loc[:, : 10])
    

    col2.subheader('**After Data Splitting & Scaling**')
    X_train = pd.DataFrame(X_train)
    col2.markdown('Training Set')
    col2.info(X_train.shape)
    X_test = pd.DataFrame(X_test)
    col2.markdown('Test Set')
    col2.info(X_test.shape)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    optional1 = col1.expander('Learning Parameters',False)
    with optional1:
        parameter_n_estimators = optional1.slider('Number of estimators (n_estimators)',0,1000,100,100)
        parameter_max_features = optional1.slider('Max features (max_featuures)',0,15,6,1)
        parameter_min_samples_split = optional1.slider('Minimum number of samples required to split an internal node (min_samples_split)',1,10,2,1)
        parameter_min_samples_leaf = optional1.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)',1,10,1,1)

    optional2 = col1.expander('General Parameters',False)
    with optional2:
        parameter_random_state = optional2.slider('Seed number (random_state)',0,1000,42,1)
        parameter_bootstrap = optional2.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = optional2.select_slider('Whether to use out-of-bag samples (oob_score)', options=[False,True])
        parameter_n_jobs = optional2.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1,-1])   

    rfc = RandomForestClassifier(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf, 
        bootstrap=parameter_bootstrap, 
        oob_score=parameter_oob_score, 
        n_jobs=parameter_n_jobs)
    rfc.fit(X_train,y_train) 

    col3.subheader('**Model Performance**')

    # Testing the model
    y_pred_rfc = rfc.predict(X_test)
    col3.markdown('Confusion Metrix')
    cm = confusion_matrix(y_test,y_pred_rfc)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10,6))
    ax=sns.heatmap(cm,annot=True,fmt='d')
    col3.pyplot(fig)

    accuracy = metrics.accuracy_score(y_pred_rfc, y_test)*100
    precision = metrics.precision_score(y_pred_rfc,y_test)*100
    recall = metrics.recall_score(y_pred_rfc,y_test)*100
    r1_score  = metrics.f1_score(y_pred_rfc,y_test)*100   

    v1 = pd.DataFrame({'Parameters': 'Accuracy Score'
        , 'Values' : "{:.2f}".format(accuracy)},index={'1'})

    v2 = pd.DataFrame({'Parameters': 'Precision Score'
        , 'Values' : "{:.2f}".format(precision)},index={'2'})

    v3 = pd.DataFrame({'Parameters': 'Recall Score'
        , 'Values' : "{:.2f}".format(recall)},index={'3'})

    v4 = pd.DataFrame({'Parameters': 'R1 Score'
        , 'Values' : "{:.2f}".format(r1_score)},index={'4'})

    result = pd.concat([v1,v2,v3,v4])
    result.columns = ['Parameters','Values']
    col3.write(result)

    col3.subheader('**ROC Curve**')
    plt.style.use('dark_background')
    plt.figure(figsize=(8,5))
    # Computing False postive rate, and True positive rate
    fpr,tpr,threshold=roc_curve(y_test,y_pred_rfc)
    # Calculating Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,y_pred_rfc)
    # Now, plotting the computed values
    plt.plot(fpr, tpr,label = "RF" , color="red", linewidth=2)
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    col3.pyplot(plt)

    optional3 = col1.expander('Grid Parameters',False)
    with optional1:
        grid_parameter_n_estimators =  optional3.slider('Number of estimators (n_estimators)',0,500,(10,70),50)
        grid_parameter_n_estimators_step =  optional3.number_input('Step size for n_estimators',10)
        grid_parameter_max_features =  optional3.slider('Max Features (max_features)',1,50,(1,6),1)
        optional3.number_input('Step size for max_features',1)
        n_estimators_range = np.arange(grid_parameter_n_estimators[0], grid_parameter_n_estimators[1]+grid_parameter_n_estimators_step, grid_parameter_n_estimators_step)
        max_features_range = np.arange(grid_parameter_max_features[0], grid_parameter_max_features[1]+1, 1)
        grid_parameter_n_jobs =  optional3.select_slider('n_jobs', options=[1,-1])
        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)   

    col2.subheader('**After Hyperparameter tuning**')
    grid = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)
    y_pred_test = grid.predict(X_test)
    col2.write('R2-Score')
    col2.info(r2_score(y_test, y_pred_test))

    col2.write('MSE')
    col2.info(mean_squared_error(y_test,y_pred_test)) 

    col2.markdown('Best Parameters & Best Score') 

    p1 = grid.best_params_['n_estimators']
    p2 = grid.best_params_['max_features']

    r1 = pd.DataFrame({'Parameters': 'Accuracy', 'Values' : "{:.2f}".format((grid.best_score_)*100)},index={'1'})
    r2 = pd.DataFrame({'Parameters': 'Best Parameters', 'Values' : "'{}': {} {} '{}': '{}'".format('n_estimators',p1,",",'max_features',p2)},index={'2'})
        
    res = pd.concat([r1,r2])
    res.columns = ['Parameters','Values']
    col2.write(res)

    col2.subheader('**Parameters**')
    col4,col5,col6 = st.columns((1,1,1))
    col4.markdown('Model Parameters')
    col4.write(rfc.get_params())
    col5.markdown('Grid Parameters')
    col5.write(grid.get_params())

    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features','n_estimators','R2'] 
    grid_pivot = grid_reset.pivot('max_features','n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
            
                
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))

    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter Tuning Plot',
                    scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='R2'),
                        autosize=False,
                        width=800, height=800,
                        margin=dict(l=65, r=50, b=65, t=90))    

    col6.plotly_chart(fig)  

    
    






    


