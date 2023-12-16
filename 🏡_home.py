import streamlit as st

from layouts.mainlayout import mainlayout  



@mainlayout
def home():

    st.write("LegalEase is a legal research tool that uses the latest advances in AI to help you find relevant case law.")

    with st.expander("How does it work?"):
        st.info("LegalEase uses a state-of-the-art language model to rephrase your question in a way that is more likely to match the language used in the case law. It then uses a search engine to find the most relevant cases and displays them in order of relevance.", icon="ℹ️")


home()