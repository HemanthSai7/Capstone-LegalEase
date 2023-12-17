import streamlit as st

from layouts.mainlayout import mainlayout


@mainlayout
def home():
    st.write(
        "LegalEase is a legal research tool that uses the latest advances in AI to help you find relevant case law."
    )

    with st.expander("How does it work?", expanded=True):
        st.info(
            "LegalEase uses a state-of-the-art language model to rephrase your question in a way that is more likely to match the language used in the case law. It then uses a search engine to find the most relevant cases and displays them in order of relevance.",
            icon="ℹ️",
        )

    with st.expander("What is a language model?", expanded=True):
        st.info(
            "A language model is a machine learning model that is trained to predict the next word in a sentence. The latest language models are trained on huge amounts of text and can generate text that is indistinguishable from human writing.",
            icon="ℹ️",
        )

    st.divider()

    st.markdown(
        "<h2 style='text-align: center; color: black;'>LegalEase Architecture</h1>",
        unsafe_allow_html=True,
    )
    st.image("results/architecture.png")


home()
