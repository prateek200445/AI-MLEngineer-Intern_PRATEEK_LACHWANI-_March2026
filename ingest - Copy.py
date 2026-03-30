from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Hard-coded config
PDF_FOLDER = "data_pdfs"
QDRANT_URL = "http://20.187.144.184:6333"
QDRANT_COLLECTION = "pdf_db"

# -------------------------
# Load PDFs
# -------------------------
all_documents = []
for file_name in os.listdir(PDF_FOLDER):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_FOLDER, file_name)
        print(f"Loading: {file_name}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

print(f"Total PDFs loaded: {len(all_documents)}")

# -------------------------
# Split text into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=350
)
texts = text_splitter.split_documents(all_documents)
print(f"Total text chunks created: {len(texts)}")

# -------------------------
# Load embeddings
# -------------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embeddings loaded")

# -------------------------
# Connect to Qdrant and push data
# -------------------------
qdrant = Qdrant.from_documents(
    documents=texts,          # <-- pass as 'documents'
    embedding=embeddings,
    url=QDRANT_URL,
    prefer_grpc=False,
    collection_name=QDRANT_COLLECTION
)


print("Qdrant loaded and data ingested successfully")
