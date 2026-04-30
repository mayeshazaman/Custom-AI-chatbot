import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import google.generativeai as genai
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()
genai.configure(api_key=api_key)

FAISS_INDEX_PATH = "faiss_index"

# ── PDF helpers ───────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs: list) -> str:
    """Extract raw text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
    return splitter.split_text(text)


def build_vector_store(chunks: list[str]) -> None:
    """Embed chunks and persist a FAISS index to disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local(FAISS_INDEX_PATH)


# ── QA chain ─────────────────────────────────────────────────────────────────
from langchain_core.output_parsers import StrOutputParser

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "This question is out of scope of the uploaded documents."
    Never make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return prompt | model | StrOutputParser()


def answer_question(user_question: str) -> str:
    if not os.path.exists(FAISS_INDEX_PATH):
        return "⚠️ No index found. Please upload and process PDFs first."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=4)

    if not docs:
        return "No relevant content found. Try rephrasing or upload more relevant PDFs."

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    return chain.invoke({"context": context, "question": user_question})


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📄",
        layout="wide",
    )
    st.header("📄 Chat with your PDFs using Gemini")
    st.caption("Upload one or more PDFs, process them, then ask anything about their content.")

    # Initialise chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role": ..., "content": ...}

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📂 Document Manager")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Submit & Process", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Extracting text and building index…"):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded PDFs.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        build_vector_store(chunks)
                        st.success(f"✅ Processed {len(pdf_docs)} file(s) into {len(chunks)} chunks.")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown(
            "**Tips**\n"
            "- Upload multiple PDFs at once.\n"
            "- Re-process whenever you add new files.\n"
            "- Questions must relate to the uploaded content."
        )

    # ── Chat interface ────────────────────────────────────────────────────────
    # Render existing history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # New question input
    user_question = st.chat_input("Ask a question about your PDF files…")
    if user_question:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = answer_question(user_question)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()