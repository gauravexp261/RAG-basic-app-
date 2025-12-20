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


import gradio as gr

def warn(*args, **kwarg):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

model_id = 'ibm/granite-3-2-8b-instruct'
parameters = {
GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
GenParams.MAX_NEW_TOKENS: 512} # this randomness or creativity of the model's responses


def get_llm():
    return WatsonxLLM(model_id='ibm/granite-3-2-8b-instruct',
    url="",
    project_id = "",
    params = parameters)

# a = get_llm()
# print(a.invoke("hello"))

    # "api_key": "your api key here"
    # uncomment above when running locally
