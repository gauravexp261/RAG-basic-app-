from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from config import get_llm
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings  # Backup embeddings




def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_doc = loader.load()
    return loaded_doc

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    chunks = text_splitter.split_documents(data)
    return chunks

def watsonx_embedding():
    return HuggingFaceEmbeddings()


def build_vector_db(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def retriever(file):
    data = document_loader(file)
    chunks = text_splitter(data)
    vectordb = build_vector_db(chunks)
    retriever_obj  = vectordb.as_retriever()
    return retriever_obj 

def retriever_qa(file,query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm = llm, chain_type = 'stuff',
    retriever= retriever_obj , return_source_documents=False)
    response = qa.invoke(query)
    return response['result']


rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging= 'never',
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label='Answer'),
    title='📄 PDF Question Answering (RAG)',
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


rag_application.launch(server_name="127.0.0.1", server_port= 7861)
