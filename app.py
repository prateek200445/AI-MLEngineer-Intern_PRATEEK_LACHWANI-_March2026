from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

from langchain_google_genai import ChatGoogleGenerativeAI


from config import settings


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ------------------ EMBEDDINGS (BGE – KEPT) ------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": settings.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE}
)

# ------------------ QDRANT CONNECTION ------------------
url = settings.QDRANT_URL or f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

client = QdrantClient(
    url=url,
    prefer_grpc=settings.PREFER_GRPC,
    api_key=settings.QDRANT_API_KEY
)

print("✅ Qdrant client connected")

db = Qdrant(
    client=client,
    collection_name=settings.QDRANT_COLLECTION,
    embeddings=embeddings   # ← KEEP THIS (important for your version)
)

print("✅ Qdrant vector store ready")

# ------------------ SEARCH ------------------
def search_docs(query: str):
    return db.similarity_search_with_score(query, k=5)

# ------------------ PROMPT ------------------
def build_prompt(docs, query):
    if not docs:
        return f"No relevant context found. Answer the question:\n{query}"

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc, _ in docs
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.format(context=context_text, question=query)

# ------------------ LLM ------------------
def get_llm():
    provider = settings.LLM_PROVIDER.lower()

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=settings.GENERATIVE_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=1024
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ------------------ ASK LLM ------------------
def ask_llm(prompt: str):
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content
