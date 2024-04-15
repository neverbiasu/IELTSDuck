# import libs
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain.vectorstores import Chroma as Vectorstore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from BCEmbedding import EmbeddingModel
from pathlib import Path
from pprint import pprint
from LLM import InternLM

# Load data
file_path = Path('./data/data_processed.json') 
json_data = json.loads(Path(file_path).read_text())
# pprint(data)

# Split data
chunk_size = 50
splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
docs = splitter.create_documents(json_data)

# Embedding model
embeddings = HuggingFaceEmbeddings()
# embeddings = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")
# texts = [doc.page_content for doc in docs]
# embeddings = embedding_model.encode(texts)
# pprint("len of embeddings, and len of embedding[0]", len(embeddings), len(embeddings[0]))

# Indexing
# faiss_index = FAISS.from_documents(pages, embeddings)
persist_directory = './data_base/vector_db/chroma'
vectordb = Vectorstore.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory)
vectordb.persist()

# Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

# Load model
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# Inferencing
model = model.eval()
# response, history = model.chat(tokenizer, "hello", history=[])
# print(response)
# # Output: Hello? How can I help you today?
# response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
# print(response)

while True:
    query = input("请输入查询: ")  # Get user input from console
    if query.lower() == 'exit':
        print("退出程序。")
        break

    retrieved_docs = cache_retriever.retrieve(query)

    # Use LLM to generate response based on retrieved docs
    context = " ".join([doc.page_content for doc in retrieved_docs[:5]])  # Limit context size to 5 docs
    response = model.generate(tokenizer(context, return_tensors='pt').input_ids.to(model.device), max_length=512)
    print("回答:", tokenizer.decode(response[0], skip_special_tokens=True))