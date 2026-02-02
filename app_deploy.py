"""
app_deploy.py

Purpose:
- Streamlit Cloud deployment version of Research-RagChat
- Uses Streamlit Secrets for OPENAI_API_KEY
- Recruiter-facing, zero setup required
- Chrome-safe PDF handling (download instead of iframe)

Do NOT use this file for local .env-based development.
"""

import os
import streamlit as st

# ------------------ API KEY (DEPLOYMENT SAFE) ------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not configured. Please set it in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------ LANGCHAIN IMPORTS ------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------------ CONFIG ------------------
PDF_PATH = "research.pdf"
MAX_QUERIES = 20

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="RAG Research Assistant",
    layout="wide"
)

st.title("📄 RAG Research Paper Assistant")

st.markdown("""
**How to use**
1. Download and read the research paper
2. Ask questions in the chat panel
3. Answers are strictly grounded in the document

⚠️ If the information is not present in the paper, the assistant will reply:
**_Not found in the document._**
""")

# ------------------ QUERY LIMIT (API PROTECTION) ------------------
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ------------------ LAYOUT ------------------
left, right = st.columns(2)

# ------------------ PDF ACCESS (CHROME SAFE) ------------------
with left:
    st.subheader("📘 Research Paper")

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="⬇️ Download Research Paper (PDF)",
        data=pdf_bytes,
        file_name="research.pdf",
        mime="application/pdf",
    )

    st.info(
        "For security reasons, Chrome blocks embedded PDFs on hosted apps. "
        "Please download or open the PDF in a new tab while using the assistant."
    )

# ------------------ VECTOR STORE ------------------
@st.cache_resource
def build_vectorstore():
    with st.spinner("Indexing research paper..."):
        pages = PyPDFLoader(PDF_PATH).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ------------------ LLM ------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ------------------ PROMPT ------------------
prompt = PromptTemplate.from_template("""
You are a research assistant.

Answer the question using ONLY the provided context.
You may summarize, combine, and paraphrase information.
If the answer cannot be derived from the context, reply exactly:

Not found in the document.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------ CHAT UI ------------------
with right:
    st.subheader("💬 Ask Questions")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask something about the research paper")

    if query:
        if st.session_state.query_count >= MAX_QUERIES:
            st.warning("Demo query limit reached. Please refresh the page.")
            st.stop()

        st.session_state.query_count += 1

        st.chat_message("user").write(query)

        with st.spinner("Generating grounded answer..."):
            answer = rag_chain.invoke(query)
            sources = retriever.invoke(query)

        st.chat_message("assistant").write(answer)

        with st.expander("📚 Source Evidence"):
            for doc in sources:
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Page {page}**")
                st.write(doc.page_content)

        st.session_state.chat.append({"role": "user", "content": query})
        st.session_state.chat.append({"role": "assistant", "content": answer})
