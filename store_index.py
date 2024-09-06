from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain.vectorstores import Chroma

extracted_data = load_pdf(r"C:\Users\hs081\OneDrive\Desktop\Medical Bot\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()


persist_directory = "VectorDataBase"
vectordb  = Chroma.from_documents(documents=text_chunks,
                                  embedding = embeddings,
                                  persist_directory = persist_directory)


vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory = persist_directory ,
                  embedding_function = embeddings)

retriever = vectordb.as_retriever()
