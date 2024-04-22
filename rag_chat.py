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
file_path = Path('./data/train_processed.json') 
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
vectordb = Vectorstore(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)

# Retriever
# retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

# Load model
llm = InternLM()
llm.predict("你是谁")

# Prompt template
template = """你是一个雅思作文小助手，需要帮用户按照雅思官方标准批改（打分+评价）他们的作文。参考以下上下文为模板来批改用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道。
有用的回答:"""

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

# Chat
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

# 检索问答链回答效果
question = """Question: Interviews form the basic criteria for most large companies. However, some people think that the interview is not a reliable method of choosing whom to employ and there are other better methods. To what extent do you agree or disagree? Essay: It is believed by some experts that the traditional approach of recruiting candidates which is interviewing is the best way, whereas others think different methods such as exams writing, CVs, cover letters or application letters and many more are good. I strongly agree with the statement, "interview is the most reliable approach to recruit workers" because this method assists the recruiters to know the person and his ability to do the work and their problem-solving abilities. 

To begin with, an interview enables the  recruiter to know the kind of person he or she is recruiting. It helps the employer to see the personality traits of the employee such as how he answers questions, his facial mannerisms and also his communication skills, that is, whether introvert or extrovert, also his teamwork skill is measured during the dialogue. For instance, jobs like sales personnel require good communication skills to be able to do the work effectively and efficiently. So interviews allow the manager to assess whether or not the applicant qualifies for the job. 

Furthermore, recruiters also assess the applicant's ability to solve problems when they arise. A good idea generated or how one handles situations can bring great development to the company. For instance, pressure can put fear into an employee which can make him make a wrong decision that can bring loss to the company, while some too can take pressure in a calm action and make a good decision. 

On the other hand, other methods such as CVs, cover letters, the use of only certificates and many more are not a suitable step to recruit an applicant due to the fact that it does not allow the recruiter to see the full potential of the candidate. Information found in the CV or cover letter may not be true because people lie to obtain what they desire. In the same way, a candidate can also lie to acquire the position. 

To sum up, I think an interview is still the most reliable practice of hiring employees rather than using other methods. So I suggest managers use only interviews as a means of sourcing workers for their companies. 请帮我的这一片作文打分"""
result = qa_chain({"query": question})
print("检索问答链回答 question 的结果：")
print(result["result"])

# 仅 LLM 回答效果
result_2 = llm(question)
print("大模型回答 question 的结果：")
print(result_2)

torch.cuda.empty_cache()
