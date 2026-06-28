import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

# 1. System Setup & Resource Caching
load_dotenv()
st.set_page_config(page_title="Pak Industrial AI Tech Support Hub", page_icon="🏭", layout="wide")

@st.cache_resource
def initialize_system():
    """Loads the embedding model and connects to your 18,444 chunks instantly."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_connection = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # Single stable LLM instance for writing the final response
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    return db_connection, llm

db_connection, llm = initialize_system()

# 2. The Single Answer Generation Prompt
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a Senior Technical Support Specialist for Pakistan Stock Exchange industrial assets.\n"
     "Synthesize a clear, highly accurate, and professional technical or financial answer based strictly on the provided context passages.\n"
     "If the context does not contain enough data to answer, state clearly that the local manuals do not contain this information.\n"
     "Do not invent or extrapolate any data points.\n\n"
     "Context Passages:\n{context}"),
    ("human", "Question: {question}")
])
generator_chain = generator_prompt | llm | StrOutputParser()

# 3. User Interface Design
st.title("🏭 Pakistan Industrial Giants Tech Support Hub")
st.caption("High-Performance Semantic Search & Corporate Intelligence Network")

with st.sidebar:
    st.header("⚙️ Filter Controls")
    ui_company = st.selectbox(
        "Target Sector Filter:",
        options=["automated", "hubco", "engro", "sazgar", "ogdc", "fauji", "lucky", "systems"],
        format_func=lambda x: "🧠 Scan All Folders" if x == "automated" else f"🏢 {x.upper()}"
    )

user_query = st.text_input("Enter your technical or financial query:", placeholder="e.g., Engro cash flow or production metrics...")

if user_query:
    # STEP 1: Execute Local Semantic Search First (Uses 0 Tokens)
    with st.status("🔍 Searching Local Database...", expanded=True) as status_box:
        st.write("Reading your pre-compiled vector database index...")
        
        # Build metadata filter mapping
        search_filter = {} if ui_company == "automated" else {"company": ui_company}
        
        # Pull top 4 closest matching text blocks from your hard drive
        retrieved_chunks = db_connection.similarity_search(
            query=user_query,
            k=4,
            filter=search_filter
        )
        
        if not retrieved_chunks:
            status_box.update(label="❌ No matching documents found.", state="error")
            st.error("Could not locate any matching text pieces inside the local database index.")
        else:
            status_box.write(f"✅ Successfully extracted {len(retrieved_chunks)} relevant source passages.")
            for idx, doc in enumerate(retrieved_chunks):
                status_box.write(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 Passage [{idx+1}] File Source: *{doc.metadata.get('source_file')}*")
            
            # Compile the raw text blocks together
            context_block = "\n\n---\n\n".join([doc.page_content for doc in retrieved_chunks])
            
            # STEP 2: Make exactly ONE API call to generate the final answer
            status_box.write("🤖 Transmitting passages to Gemini for professional synthesis...")
            try:
                final_answer = generator_chain.invoke({
                    "context": context_block,
                    "question": user_query
                })
                status_box.update(label="✅ Analysis Finalized Successfully", state="complete")
                
                # Render the final answer beautifully on screen
                st.markdown("### 📝 Grounded Technical Solution Report")
                st.write(final_answer)
                
            except Exception as api_err:
                status_box.update(label="⚠️ Cloud API Congestion", state="error")
                st.error(f"Google API Error: {api_err}")
