from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Streamlit UI
st.header("🎥 YouTube Video Summarizer")
user_input = st.text_input("Paste YouTube Video Link")

# Extract video ID
if "v=" in user_input:
    video_id = user_input.split("v=")[-1].split("&")[0]
else:
    video_id = user_input

# Transcript
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])
    transcript = " ".join([chunk["text"] for chunk in transcript_list])
except TranscriptsDisabled:
    st.error("⚠️ No captions available for this video.")
    transcript = ""

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
chunks = splitter.create_documents([transcript])

# Embeddings + Vector DB
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Format docs
def format_docs(retrieved_docs):
    return " ".join(doc.page_content for doc in retrieved_docs)

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant that summarizes YouTube videos.

    Context:
    {context}

    Question:
    {question}

    Answer in clear, concise paragraphs:
    """
)

# Chains
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

final_chain = parallel_chain | prompt | model | parser

# Run
if st.button("Summarize"):
    summary = final_chain.invoke("Summarize this video")
    st.write("## 📝 Summary")
    st.write(summary)
