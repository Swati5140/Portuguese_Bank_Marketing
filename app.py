import streamlit as st
from multiapp import MultiApp
from apps import home, model, pred

app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Model Customization", model.app)
app.add_app("Prediction", pred.app)


app.run()