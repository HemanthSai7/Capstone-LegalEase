from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import transformers
import os
from llama_index.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

upload_dir = "data"
# List the files in the directory
files = os.listdir(upload_dir)
# Print the list of files
print("List of files:")
for file in files:
    print(file)

documents = SimpleDirectoryReader(upload_dir).load_data()


# llm = LlamaCPP(
#     # You can pass in the URL to a GGML model to download it automatically
#     model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
#     # optionally, you can set the path to a pre-downloaded model instead of model_url
#     model_path=None,
#     temperature=0.1,
#     max_new_tokens=256,
#     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
#     context_window=3900,
#     # kwargs to pass to __call__()
#     generate_kwargs={},
#     # kwargs to pass to __init__()
#     # set to at least 1 to use GPU
#     model_kwargs={"n_gpu_layers": -1},
#     # transform inputs into Llama2 format
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

print("Embedding Model")
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="BAAI/llm-embedder"))

print("Reached Service Context")
service_context = ServiceContext.from_defaults(
    chunk_size=256, llm=OpenAI(model="gpt-3.5-turbo"), embed_model=embed_model
)

print("Reached Index")
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

print("Reached Query Engine")
query_engine = index.as_query_engine()

while True:
    print("Enter your query: ")
    query = input()
    response = query_engine.query(query)
    print(response)
