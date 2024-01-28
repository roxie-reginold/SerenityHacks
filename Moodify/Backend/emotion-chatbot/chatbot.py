import langchain_helper as lch
import streamlit as st

st.title("Speak to our chatbot, Moodie")

emotion = st.text_input("How are you feeling?")

if emotion:
    scale = st.selectbox("On a scale of 1-10, to what extent do you feel upset?", range(1, 11))
    response = lch.generate_result(emotion, scale)

    st.write("Chatbot Response:", response)
