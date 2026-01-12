import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import base64

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("📄 RAG Research Paper Assistant")

# ---------------- PDF Viewer ----------------
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px"></iframe>',
        unsafe_allow_html=True
    )

left, right = st.columns([1,1])
with left:
    st.subheader("Research Paper")
    show_pdf("research.pdf")

# ---------------- Load CPU LLM ----------------
@st.cache_resource
def load_llm():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ---------------- Build High-Quality RAG ----------------
@st.cache_resource
def build_rag():
    pages = PyPDFLoader("research.pdf").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = FAISS.from_documents(chunks, embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a research assistant.
Answer ONLY from the context below.
If the answer is not present, say: Not found in the document.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 12}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

qa = build_rag()

# ---------------- Chat UI ----------------
with right:
    st.subheader("Chat with the Paper")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask something about the research paper")

    if query:
        st.chat_message("user").write(query)
        result = qa(query)

        answer = result["result"]
        sources = result["source_documents"]

        st.chat_message("assistant").write(answer)

        with st.expander("📚 Source text used"):
            for doc in sources:
                st.markdown(f"**Page {doc.metadata['page']}**")
                st.write(doc.page_content)

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})
