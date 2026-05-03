"""
FastAPI backend for PDF chatbot.
Run: uvicorn app:app --reload --port 8000
"""

import os
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import defaultdict
import logging

import google.generativeai as genai
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=api_key)

FAISS_INDEX_PATH = "faiss_index"

app = FastAPI(title="PDF Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_store = defaultdict(list)

import logging

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),   # saves to file
        logging.StreamHandler()                 # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)
# ── PDF helpers ───────────────────────────────────────────────────────────────
def get_pdf_text(pdf_bytes_list: list[bytes]) -> str:
    text = ""
    for data in pdf_bytes_list:
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
    return splitter.split_text(text)


def build_vector_store(chunks: list[str]) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local(FAISS_INDEX_PATH)


# ── QA chain ──────────────────────────────────────────────────────────────────
def get_conversational_chain():
    prompt_template = """
    You are a helpful chatbot. Answer using only the provided context.
    If the answer is not in the context, say:
    "This question is outside scope of the uploaded documents."
    Never make up information.

    Context:
    {context}

    Conversation so far:
    {history}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "history", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return prompt | model | StrOutputParser()


# ── Validation ────────────────────────────────────────────────────────────────
def compute_confidence(answer: str, docs: list) -> float:
    """Word overlap between answer and retrieved chunks, ignoring stopwords."""
    stopwords = {"the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "was", "are", "for"}
    context = " ".join([doc.page_content for doc in docs]).lower()
    words = set(answer.lower().split()) - stopwords
    if not words:
        return 0.0
    matched = sum(1 for w in words if w in context)
    return round(matched / len(words), 2)


def validate_answer(answer: str, docs: list) -> dict:
    """Check if the answer is grounded in the docs and score its confidence."""
    out_of_scope = any(p in answer.lower() for p in ["outside scope", "not in the provided context", "i don't know"])
    confidence = compute_confidence(answer, docs)

    if out_of_scope or confidence < 0.3:
        return {
            "answer": "I couldn't find a confident answer in the uploaded documents.",
            "confidence": confidence,
            "status": "low_confidence",
        }

    return {
        "answer": answer,
        "confidence": confidence,
        "status": "ok",
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────
def answer_question(user_question: str, session_id: str) -> dict:
    if not Path(FAISS_INDEX_PATH).exists():
        return {"answer": "No index found. Please upload PDFs first.", "confidence": 0.0, "status": "no_index"}

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=4)

    if not docs:
        return {"answer": "No relevant content found.", "confidence": 0.0, "status": "no_docs"}

    # Format history as a readable string for the prompt
    history = conversation_store[session_id]
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history[-6:]  # last 6 messages = 3 exchanges
    )

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    raw_answer = chain.invoke({
        "context": context,
        "history": history_text or "None yet.",
        "question": user_question,
    })

    result = validate_answer(raw_answer, docs)

    # Save to history only if answer was good
    if result["status"] == "ok":
        conversation_store[session_id].append({"role": "user", "content": user_question})
        conversation_store[session_id].append({"role": "assistant", "content": result["answer"]})

    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided.")

    pdf_bytes_list = [await f.read() for f in files]
    raw_text = get_pdf_text(pdf_bytes_list)

    if not raw_text.strip():
        raise HTTPException(422, "No readable text found in the uploaded PDFs.")

    chunks = get_text_chunks(raw_text)
    build_vector_store(chunks)

    return {"message": f"Processed {len(files)} file(s) into {len(chunks)} chunks. Index is ready."}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"  # frontend sends this, defaults to "default"

@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    logger.info(f"Upload request received — {len(files)} file(s)")
    if not files:
        raise HTTPException(400, "No files provided.")

    pdf_bytes_list = [await f.read() for f in files]
    raw_text = get_pdf_text(pdf_bytes_list)

    if not raw_text.strip():
        logger.warning("Upload failed — no readable text found in PDFs")
        raise HTTPException(422, "No readable text found in the uploaded PDFs.")

    chunks = get_text_chunks(raw_text)
    build_vector_store(chunks)
    logger.info(f"Index built successfully — {len(chunks)} chunks from {len(files)} file(s)")

    return {"message": f"Processed {len(files)} file(s) into {len(chunks)} chunks. Index is ready."}
@app.post("/ask")
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    logger.info(f"Question received | session={req.session_id} | q={req.question[:60]}")
    result = answer_question(req.question, req.session_id)
    logger.info(f"Answer sent | status={result['status']} | confidence={result['confidence']}")
    return result

@app.delete("/clear/{session_id}")
async def clear_history(session_id: str):
    conversation_store.pop(session_id, None)
    logger.info(f"Conversation cleared | session={session_id}")
    return {"message": f"Conversation cleared for session {session_id}."}

@app.get("/health")
async def health():
    index_ready = Path(FAISS_INDEX_PATH).exists()
    logger.info(f"Health check | index_ready={index_ready}")
    return {"status": "ok", "index_ready": index_ready}

    