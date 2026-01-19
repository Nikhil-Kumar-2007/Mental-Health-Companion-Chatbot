import streamlit as st

from llm import ai_assistant_reply


st.set_page_config(
    page_title="Mental Health Companion Chatbot",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align:center;'>🧠 Mental Health Companion Chatbot</h1>",
    unsafe_allow_html=True
)


if "chat" not in st.session_state:
    st.session_state.chat = []

user_msg = st.chat_input("Hi! I am your friend 😊")    

if user_msg:
    st.session_state.chat.append(('user', user_msg))
    bot_msg = ai_assistant_reply(user_msg)
    st.session_state.chat.append(('assistant',  bot_msg))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)   

   
