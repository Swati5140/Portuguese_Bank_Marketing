from multiapp import MultiApp
import streamlit as st
from PIL import Image
import base64
import pandas as pd
import numpy as np


img = Image.open("icon.png")
st.set_page_config(page_title='Bank Term Policy Subscription Prediction App', page_icon=img, layout='wide')

def app():
    hide_menu_style = """
            <style>
            #MainMenu {visibility:hidden; }
            footer {visibility:hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.title("Bank Term Subscription Prediction App")
    image=Image.open("bank_marketing.png")
    st.image(image,use_column_width=True)

    st.write("""
    This App predicts the subscription for a term deposit by the existing clients.
    As a result Portuguese Bank can focus marketing efforts on such clients to increase the revenue.

    **Python Libraries Used : ** streamlit, pandas, numpy, sklearn
    
    
    """)
    st.write("**Dataset : ** Data is obtained from Kaggle.")

    df = pd.read_csv('C:/Users/iamsw/OneDrive/Desktop/Data Science/Projects/PBM App/apps/bank-additional-full.csv',sep=';')

    def convert_df(df):
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(df)

    st.download_button(
        "Download Data as CSV",
        csv,
        "bank_data.csv",
        "text/csv",
        key='bank-additional-full'
    )


    

