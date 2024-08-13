from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
# from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import uuid
from data_ingestion import pdfProcess

app = Flask(__name__)

folder_path = "db"
device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)

def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
       
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# cached_llm = Ollama(model="llama3")

cached_llm = llm_pipeline()
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/api/documents/process", methods=["POST"])
def pdfPost():
    response_dict = {}
    pdfProcess(response_dict)
    return jsonify(response_dict)
    # return jsonify({"message": "Documents processed and embeddings created successfully."})

@app.route("/api/chat/start", methods=["POST"])
def startChat():
    json_content = request.json
    asset_id = json_content.get("asset_id")






def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()