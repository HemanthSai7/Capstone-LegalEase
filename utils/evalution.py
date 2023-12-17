from .models import *

from sentence_transformers import util

def evaluator(base_response, *responses):
    """
    Evaluates the responses of the models and returns the best one.
    """

    embeddings = load_embeddings()

    base_response_embedding = embeddings.encode(
        base_response, normalize_embeddings=True
    )

    encoded_responses = embeddings.encode(responses, normalize_embeddings=True)

    dot_product_similarity = encoded_responses @ base_response_embedding.T
    cosine_similarity = util.pytorch_cos_sim(encoded_responses, base_response_embedding)

    dot_product_index = dot_product_similarity.argmax()
    cosine_similarity_index = cosine_similarity.argmax().item()

    return (
        responses[cosine_similarity_index],
        cosine_similarity[cosine_similarity_index],
        dot_product_similarity[dot_product_index],
        cosine_similarity_index,
        dot_product_similarity,
        cosine_similarity,
    )