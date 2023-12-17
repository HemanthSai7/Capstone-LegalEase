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

    response, score, index, similarity = evaluator(
        response_gpt3, response_mistral, response_falcon, response_llama2
    )

    # st.write(f"Reponse: {response_mistral}")
    # st.write(f" handler: {retrieval_handler}")
    models = ["GPT3", "Mistral", "Falcon", "Llama2"]
    with st.chat_message(f"{models[index]}"):
        st.write(models[index])
        st.write(f"Score: {score}")
        st.write(f"Index: {index}")
        st.write(f"Reponse: {response}")
        # similarity = np.array(similarity).reshape(1, 3)
        df = pd.DataFrame(similarity, columns=["GPT3","Mistral", "Falcon", "Llama2"])
        st.scatter_chart(
            df, x="GPT3", y=["Mistral", "Falcon", "Llama2"], use_container_width=True
        )
