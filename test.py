import openai
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Test the API with a prompt completion
try:
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    model_engine = "text-davinci-codex-002"

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    st.write(response.choices[0].text.strip())

except Exception as e:
    st.error(f"An error occurred with the OpenAI API: {str(e)}")
