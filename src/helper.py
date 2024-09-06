
from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

## Extract Data From Pdf file 

def load_pdf(data):
    loader = DirectoryLoader(data , 
                            glob = "*.pdf",
                            loader_cls = PyPDFLoader)
    documents = loader.load()

    return documents


## Text Chunks 

def text_split(extracted_data):
    text_splliter = RecursiveCharacterTextSplitter(chunk_size = 500,  chunk_overlap = 50)
    text_chunks = text_splliter.split_documents(extracted_data)

    return text_chunks


## Download Embedding Model

def download_hugging_face_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding