import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

st.set_page_config(page_title="Upload and Ask", page_icon="📄")
st.title("📄 PDF Q&A with Hugging Face + LangChain")

# -------------------------
# 1. Ensure folders exist
# -------------------------
os.makedirs("data", exist_ok=True)

# -------------------------
# 2. Upload PDF
# -------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    # -------------------------
    # 3. Load PDF (cached)
    # -------------------------
    @st.cache_resource(show_spinner=True)
    def load_docs(_pdf_path):
        loader = PyPDFLoader(_pdf_path)
        return loader.load()

    documents = load_docs(pdf_path)
    st.write(f"Loaded {len(documents)} pages.")

    # -------------------------
    # 4. Create Vector Store (cached)
    # -------------------------
    @st.cache_resource(show_spinner=True)
    def create_vectorstore(_documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(_documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    vectorstore = create_vectorstore(documents)
    st.write("Vector store ready ✅")

    # -------------------------
    # 5. Setup QA Chain (cached)
    # -------------------------
    @st.cache_resource(show_spinner=True)
    def get_qa_chain(_vectorstore):
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_length=512
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=_vectorstore.as_retriever()
        )
        return qa_chain

    qa_chain = get_qa_chain(vectorstore)
    st.write("QA chain ready ✅")

    # -------------------------
    # 6. Ask questions interactively
    # -------------------------
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
        st.markdown(f"*Answer:* {answer}")