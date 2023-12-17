import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from utils import configure_retriever, PrintRetrievalHandler, StreamHandler
from utils import *


st.set_page_config(page_title="LegalEase", page_icon="⚖️")
st.title("⚖️ LegalEase")

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
gpt3_memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)
mistral_memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)
falcon_memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)
llama_memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

qa_chain_gpt3_5_turbo = load_model_gpt3_5_turbo(retriever, gpt3_memory)
qa_chain_mistral = load_model_mistral(retriever, mistral_memory)
qa_chain_falcon = load_model_falcon(retriever, falcon_memory)
qa_chain_llama2 = load_model_mistral(retriever, llama_memory)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    retrieval_handler = PrintRetrievalHandler(st.container())
    stream_handler = StreamHandler(st.empty())

    response_gpt3 = qa_chain_gpt3_5_turbo.run(
        user_query, callbacks=[retrieval_handler, stream_handler]
    )
    response_mistral = qa_chain_mistral.run(
        user_query, callbacks=[retrieval_handler, stream_handler]
    )
    response_falcon = qa_chain_falcon.run(
        user_query, callbacks=[retrieval_handler, stream_handler]
    )
    response_llama2 = qa_chain_llama2.run(
        user_query, callbacks=[retrieval_handler, stream_handler]
    )

    (
        response,
        cosine_similarity_score,
        dot_product_score,
        index,
        dot_product_similarity,
        cosine_similarity,
    ) = evaluator(response_gpt3, response_mistral, response_falcon, response_llama2)

    models = ["Mistral", "Falcon", "Llama2"]
    with st.chat_message(f"{models[index]}"):
        st.write(models[index])
        st.write(f"Cosine Similarity: {cosine_similarity_score}")
        st.write(f"Index: {index}")
        st.write(f"Reponse: {response}")

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, dot_product_similarity[0]], color="red", label="Mistral")
        ax.plot([0, 2], [0, dot_product_similarity[1]], color="green", label="Falcon")
        ax.plot([0, 3], [0, dot_product_similarity[2]], color="blue", label="Llama2")
        ax.legend()
        ax.set_xlabel("Models")
        ax.set_ylabel("Similarity")
        ax.set_title("Cosine  Similarity")
        ax.set_ylim([0, 1])
        ax.annotate(
            f"Score: {dot_product_similarity[0]}",
            xy=( 1, dot_product_similarity[0]),
            xytext=(1, dot_product_similarity[0] + 0.1),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )
        ax.annotate(
            f"Score: {dot_product_similarity[1]}",
            xy=(2, dot_product_similarity[1]),
            xytext=(2, dot_product_similarity[1] + 0.1),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )
        ax.annotate(
            f"Score: {dot_product_similarity[2]}",
            xy=(3, dot_product_similarity[2]),
            xytext=(3, dot_product_similarity[2] + 0.1),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        st.pyplot(fig)
