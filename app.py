import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Define color palette
primary_color = "#007BFF"
secondary_color = "#FFC107"
background_color = "#F8F9FA"
sidebar_background = "#2C2F33"
text_color = "#212529"
sidebar_text_color = "#FFFFFF"
header_text_color = "#000000"

# Apply styling
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #2C2F33 !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; font-size: 16px !important; }
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; font-weight: bold; }
    p, span, div { color: #212529 !important; }
    .stFileUploader>div>div>div>button { background-color: #FFC107; color: #000000; font-weight: bold; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📄 Build a RAG System with DeepSeek R1 & Ollama (TXT Support)")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a TXT file using the uploader below.
    2. Ask questions related to the document.
    3. The system will retrieve relevant content and provide a concise answer.
    """)

    st.header("Settings")
    st.markdown("""
    - **Embedding Model**: HuggingFace thenlper/gte-large 
    - **Retriever Type**: Similarity Search
    - **LLM**: initium/law_model
    """)

# File uploader for TXT
st.header("📁 Upload a TXT Document")
uploaded_file = st.file_uploader("Upload your TXT file here", type="txt")

if uploaded_file is not None:
    st.success("TXT file uploaded successfully! Processing...")

    # Save the uploaded file
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the TXT file
    loader = TextLoader("temp.txt", encoding="utf-8")
    docs = loader.load()

    # Split the document into chunks
    st.subheader("📚 Splitting the document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],  # Paragraphs and lines
        chunk_size=500,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings(model_name="thenlper/gte-large") #added "model_name=thenlper/gte-large" the parenthesis were there before

    # Create vector store and retriever
    st.subheader("🔍 Creating embeddings and setting up the retriever...")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Define the LLM and the prompt
    llm = Ollama(model="initium/law_model")
    prompt = """
    You are a educated and studied development plans reviewer for the City of Austin. You know all the rules and regulations of the City. 
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Provide well founded citations for your answer.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Define the document and combination chains
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        verbose=True
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=True
    )

    # Question input and response display
    st.header("❓ Ask a Question")
    user_input = st.text_input("Type your question related to the document:")

    if user_input:
        with st.spinner("Processing your query..."):
            try:
                response = qa(user_input)["result"]
                st.success("✅ Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a TXT file to start.")

