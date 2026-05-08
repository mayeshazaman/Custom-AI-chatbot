"""
FastAPI backend for Document chatbot (PDF, TXT, DOCX).
Run: uvicorn app:app --reload --port 8000
"""

import os
import io
import logging
from collections import defaultdict
from pathlib import Path

from docx import Document as DocxDocument
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found.")


FAISS_INDEX_PATH = "faiss_index"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Document Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_store = defaultdict(list)

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── File helpers ──────────────────────────────────────────────────────────────

def get_pdf_text_from_bytes(content: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(content))
    return "".join(page.extract_text() or "" for page in reader.pages)


def get_txt_text_from_bytes(content: bytes) -> str:
    """Decode plain text bytes."""
    return content.decode("utf-8", errors="ignore")


def get_docx_text_from_bytes(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
async def extract_text_from_files(files: list[UploadFile]) -> str:
    all_text = ""
    for file in files:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"{file.filename} exceeds 10MB limit.")
        
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type: {ext}")
        
        # extract immediately using the content you already have
        if ext == ".pdf":
            all_text += get_pdf_text_from_bytes(content)
        elif ext == ".txt":
            all_text += get_txt_text_from_bytes(content)
        elif ext == ".docx":
            all_text += get_docx_text_from_bytes(content)
    
    return all_text
def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    return splitter.split_text(text)


def build_vector_store(chunks: list[str]) -> None:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    if Path(FAISS_INDEX_PATH).exists():
        try:
            existing = FAISS.load_local(FAISS_INDEX_PATH, embeddings,
                                        allow_dangerous_deserialization=True)
            existing.merge_from(new_store)
            existing.save_local(FAISS_INDEX_PATH)
        except RuntimeError as e:
            # If merge fails due to dimension mismatch, recreate the index
            if "d ==" in str(e):
                logging.warning(f"FAISS index dimension mismatch. Recreating index: {e}")
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
                new_store.save_local(FAISS_INDEX_PATH)
            else:
                raise
    else:
        new_store.save_local(FAISS_INDEX_PATH)


# ── QA chain ──────────────────────────────────────────────────────────────────
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
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
        return {"answer": "No index found. Please upload documents first.", "confidence": 0.0, "status": "no_index"}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=4)

    if not docs:
        return {"answer": "No relevant content found.", "confidence": 0.0, "status": "no_docs"}

    history = conversation_store[session_id]
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history[-6:]
    )

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    raw_answer = chain.invoke({
        "context": context,
        "history": history_text or "None yet.",
        "question": user_question,
    })

    result = validate_answer(raw_answer, docs)

    if result["status"] == "ok":
        conversation_store[session_id].append({"role": "user", "content": user_question})
        conversation_store[session_id].append({"role": "assistant", "content": result["answer"]})

    return result


# ── Routes ────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    
class AnswerResponse(BaseModel):
    answer: str        # "The policy states that..."
    confidence: float  # 0.87
    status: str        # "ok"
class UploadResponse(BaseModel):
    message: str       # "Processed 2 file(s) into 45 chunks."


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    logger.info(f"Upload request received - {len(files)} file(s)")
    if not files:
        raise HTTPException(400, "No files provided.")

    raw_text = await extract_text_from_files(files)

    if not raw_text.strip():
        logger.warning("Upload failed - no readable text found")
        raise HTTPException(422, "No readable text found in uploaded files.")

    chunks = get_text_chunks(raw_text)
    build_vector_store(chunks)
    logger.info(f"Index built - {len(chunks)} chunks from {len(files)} file(s)")

    return {"message": f"Processed {len(files)} file(s) into {len(chunks)} chunks. Index is ready."}

@app.post("/ask", response_model=AnswerResponse)  # add this
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