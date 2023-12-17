import streamlit as st

from .models import *

def evaluator(base_response, *responses):
    """
    Evaluates the responses of the models and returns the best one.
    """

    embeddings = load_embeddings()

    base_response_embedding = embeddings.encode(
        base_response, normalize_embeddings=True
    )

    encoded_responses = embeddings.encode(responses, normalize_embeddings=True)

    similarity = encoded_responses @ base_response_embedding.T
    index = similarity.argmax()

    return responses[index], similarity[index], index, similarity


# kjfdhbaskjf=evaluator("Hello", "hi", "hello", "hey")
# print(kjfdhbaskjf)
