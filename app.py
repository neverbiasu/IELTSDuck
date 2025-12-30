# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.vectorstores import Chroma as Vectorstore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from modelscope import snapshot_download
from pathlib import Path
import json
import os

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## IELTSDuck")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[雅鸭](https://github.com/neverbiasu/IELTSDuck.git)"
    # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
    max_length = st.slider("max_length", 0, 1024, 512, step=1)
    system_prompt = st.text_input("System_Prompt", "现在你要是一位专业的雅思教师，请你根据我的作文进行批改。")

# 创建一个标题和一个副标题
st.title("💬 InternLM2-Chat-7B IELTSDuck")
st.caption("🚀 A streamlit chatbot powered by InternLM2 QLora")

# 定义模型路径
model_id = 'ModelE/IELTSDuck-Chat-7B'
mode_name_or_path = snapshot_download(model_id, revision='master')

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()  
    return tokenizer, model

# 加载IELTSDuck的model和tokenizer
tokenizer, model = get_model()

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])

def generate_response(prompt):
    
    # Prompt template
    template = """你是一个雅思作文小助手，需要帮用户按照雅思官方标准批改他们的作文。使用以下上下文来批改用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    如果给定的上下文无法让你做出回答，请回答你不知道。
    有用的回答:"""
    # Create a RetrievalQA instance
    qa_chain = RetrievalQA.from_chain_type(model, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": PromptTemplate(input_variables=["context","question"], template=template)})
    # Generate response
    return qa_chain.run({"question": prompt})

# Get user input
if prompt := st.chat_input():
    # Display user input
    st.chat_message("user").write(prompt)
    # Generate response
    response, _ = model.chat(tokenizer, prompt, meta_instruction=system_prompt, history=st.session_state.messages)
    # Add response to session_state messages
    st.session_state.messages.append((prompt, response))
    # Display response
    st.chat_message("assistant").write(response)
