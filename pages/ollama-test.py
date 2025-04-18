import streamlit as st
from ollama import chat
from ollama import ChatResponse
from ollama import show


st.title("➡️ Welcome to the LLM Playground")

st.divider()


user_prompt = st.text_input("Write your message")

stream = chat(
    model='deepseek-r1:14b',
    messages=[
        {
            'role': 'user',
            'content': user_prompt,
             'stream': True,
        },
    ]
)




