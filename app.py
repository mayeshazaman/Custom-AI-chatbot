"""
FastAPI backend for PDF chatbot.
Run: uvicorn backend:app --reload --port 8000
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
    "This question is outside scope of the uploaded documents."
    Never make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return prompt | model | StrOutputParser()


def answer_question(user_question: str) -> str:
    if not Path(FAISS_INDEX_PATH).exists():
        return "No index found. Please upload and process PDFs first."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=4)

    if not docs:
        return "No relevant content found. Try rephrasing or upload more relevant PDFs."

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    return chain.invoke({"context": context, "question": user_question})


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided.")

    pdf_bytes_list = []
    for f in files:
        pdf_bytes_list.append(await f.read())

    raw_text = get_pdf_text(pdf_bytes_list)
    if not raw_text.strip():
        raise HTTPException(422, "No readable text found in the uploaded PDFs.")

    chunks = get_text_chunks(raw_text)
    build_vector_store(chunks)

    return {"message": f"Processed {len(files)} file(s) into {len(chunks)} chunks. Index is ready."}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    answer = answer_question(req.question)
    return {"answer": answer}
