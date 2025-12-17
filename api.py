from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
# Import functions from app.py
from app import search_docs, build_prompt, ask_llm

api = FastAPI(
    title="RAG Search API",
    version="1.0",
    description="API for searching Qdrant + generating answers using Gemini"
)


api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    question: str


@api.get("/")
def home():
    return {"status": "RAG API is running!"}


@api.post("/query")
def query_rag(req: QueryRequest):
    try:
        question = req.question

        # 1. Search documents
        docs = search_docs(question)

        # 2. Build prompt for LLM
        final_prompt = build_prompt(docs, question)

        # 3. Ask Gemini
        answer = ask_llm(final_prompt)

        # 4. Format document details
        result_docs = [
            {
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc, score in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "documents_used": result_docs,
            "prompt_sent_to_llm": final_prompt
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
