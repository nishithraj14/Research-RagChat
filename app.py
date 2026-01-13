import os, base64
import streamlit as st
from dotenv import load_dotenv

# Force .env from THIS folder only
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded from .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "research.pdf"

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("📄 RAG Research Paper Assistant")

# ------------------ PDF Viewer ------------------
def show_pdf(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>',
        unsafe_allow_html=True
    )

left, right = st.columns(2)
with left:
    show_pdf(PDF_PATH)

# ------------------ Vector Store ------------------
@st.cache_resource
def build_vectorstore():
    pages = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ------------------ LLM ------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# ------------------ Prompt ------------------
prompt = PromptTemplate.from_template("""
You are a research assistant.

Answer the question using ONLY the provided context.
You may summarize, combine, and paraphrase information.
If the answer cannot be derived from the context, reply:

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

# ------------------ Chat UI ------------------
with right:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

    query = st.chat_input("Ask something about the research paper")

    if query:
        st.chat_message("user").write(query)

        answer = rag_chain.invoke(query)
        sources = retriever.invoke(query)

        st.chat_message("assistant").write(answer)

        with st.expander("📚 Source Evidence"):
            for d in sources:
                st.markdown(f"**Page {d.metadata.get('page', 'N/A')}**")
                st.write(d.page_content)

        st.session_state.chat.append({"role": "user", "content": query})
        st.session_state.chat.append({"role": "assistant", "content": answer})
