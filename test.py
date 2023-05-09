import openai
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

# Test the API by listing models
try:
    models = openai.Model.list()
    st.write(models)
except Exception as e:
    st.error(f"An error occurred with the OpenAI API: {str(e)}")
