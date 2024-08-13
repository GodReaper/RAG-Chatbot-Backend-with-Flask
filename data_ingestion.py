
from flask import  request, jsonify
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import uuid 

folder_path = "db"

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
def pdfProcess(response_dict):
    file = request.files["file"]

    asset_id = str(uuid.uuid4())

    file_name = file.filename
    save_file = "docs/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")
    print(f"Generated asset_id: {asset_id}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=f'{folder_path}/{asset_id}'
    )

    vector_store.persist()
    # response = {
    #     "status": "Successfully Uploaded",
    #     "filename": file_name,
    #     "doc_len": len(docs),
    #     "chunks": len(chunks),
    #     "asset_id": asset_id
    # }
    # return response
    response_dict["status"] = "Successfully Uploaded"
    response_dict["filename"] = file_name
    response_dict["doc_len"] = len(docs)
    response_dict["chunks"] = len(chunks)
    response_dict["asset_id"] = asset_id