# import libs
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma as Vectorstore
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
llm = InternLM()
llm.predict("你是谁")

# Prompt template
template = """你是一个雅思作文小助手，需要帮用户按照雅思官方标准批改他们的作文。使用以下上下文来批改用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道。
有用的回答:"""

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

