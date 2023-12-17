import pandas as pd

import streamlit as st

from layouts.mainlayout import mainlayout


@mainlayout
@st.cache_data
def results():
    st.title("📝 Results")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Cosine Similarity", "BLEU Score", "ROUGE Score", "BERT Score"]
    )
    with tab1:
        st.write(
            """
            **Metric Description**

            The cosine similarity measure computes the cosine of the angle between two vectors, which represents how similar they are in direction. Cosine similarity is a commonly used metric in natural language processing (NLP) and information retrieval to compare the similarity of text documents, as well as in other areas such as image processing and recommendation systems.
            """
        )

    with tab2:
        st.write(
            """
            **Metric Description**

            BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine’s output and that of a human: “the closer a machine translation is to a professional human translation, the better it is” – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.

            Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation’s overall quality. Neither intelligibility nor grammatical correctness are not taken into account.
            """
        )

    with tab3:
        st.write(
            """
            **Metric Description**
                
            ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

            Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.
            """
        )
    with tab4:
        st.write(
            """ 
            **Metric Description**

            BERTScore is an automatic evaluation metric for text generation that computes a similarity score for each token in the candidate sentence with each token in the reference sentence. It leverages the pre-trained contextual embeddings from BERT models and matches words in candidate and reference sentences by cosine similarity.

            Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.
            """
        )

    st.divider()
    st.write("**Here are the results of the evaluation.**")

    df = pd.read_csv("results/Evaluation_Results.csv")
    df["Bleu Score"]=df["Bleu Score"].astype(str)
    st.dataframe(df, hide_index=True)


results()
