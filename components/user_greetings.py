import streamlit as st


def user_greetings():
    with st.sidebar.expander("👋 Greetings!", expanded=True):
        st.write(
            "Welcome to LegalEase! This is a legal assistant that can help you with your legal queries. "
        )
