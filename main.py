import os
import pandas as pd
import openai
import aiofiles
import logging
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
# Define the path to save the vectorstore
VECTORSTORE_PATH = "/faiss_index"  # Adjust this path to the correct location on your Render disk

# Load vectorstore at startup
@app.on_event("startup")
def load_vectorstore():
    global vectorstore
    try:
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, HuggingFaceEmbeddings(model_name="thenlper/gte-large"), allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully.")
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise HTTPException(status_code=500, detail="Failed to load vectorstore.")

# Initialize PubMed API wrapper safely
def initialize_pubmed():
    from langchain_community.tools.pubmed.tool import PubmedQueryRun
    from langchain_community.utilities import PubMedAPIWrapper
    api_wrapper = PubMedAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    pubmed = PubmedQueryRun(api_wrapper=api_wrapper)
    return pubmed

pubmed = initialize_pubmed()

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("API-KEY"))

    def call(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content

class HistoryItem(BaseModel):
    question: str
    answer: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[HistoryItem]] = None

# Initialize tools and agent safely
@safe_execute
def initialize_tools_and_agent(pubmed, vectorstore):
    retriever_tool = Tool(
        name="retriever",
        func=vectorstore.similarity_search,
        description="Useful for retrieving documents related to the query.",
    )
    pubmed_tool = Tool(
        name="pubmed",
        func=pubmed.run,
        description="Useful for retrieving the latest research articles from PubMed.",
    )
    return retriever_tool, pubmed_tool

retriever_tool, pubmed_tool = initialize_tools_and_agent(pubmed, vectorstore)

def get_answer_from_openai(question, context):
    messages = [
        {"role": "system", "content": "You are an accomplished medical doctor, nutritionist, health psychiatrist, and healthy food chef. Provide accurate and concise answers based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    model = OpenAIEngine()
    result = model.call(messages=messages)
    return result

@app.post("/ask-question/")
def run_agentic_rag(request: QuestionRequest):
    question = request.question
    history = request.history or []

    # Retrieve contexts from the knowledge bases
    retrieved_contexts = vectorstore.similarity_search(question, k=5)
    context = " ".join([doc.page_content for doc in retrieved_contexts])

    # Add history to the context
    history_context = " ".join([f"Q: {item.question} A: {item.answer}" for item in history])

    # Enhance context with PubMed if available
    if not context:
        pubmed_response = pubmed.run(question)
        context += f" {pubmed_response}"

    # Define an enhanced question for the agent
    enhanced_question = f"""
    You are an empathetic and knowledgeable professional specializing in medicine, nutrition, psychiatry, and healthy cooking.
    When answering users' questions, prioritize accuracy and evidence-based information. Use your 'retriever' tool for accessing internal knowledge and 'pubmed' for the latest research to support your responses. Answer only the questions asked with concise and relevant information, ensuring all responses are based on tested and accurate medical knowledge.
    Start by addressing the user's concern empathetically, for example, "I'm sorry to hear that you're experiencing severe back pain." Ask necessary follow-up questions one at a time to gather sufficient information before providing advice, continuing this process until enough information is obtained to give a complete and accurate response.
    For instance, if a user mentions experiencing severe back pain recently, you might respond: "I'm sorry to hear that you're experiencing severe back pain. How long have you been experiencing this pain?" Once the user answers, based on their response, ask the next relevant question, such as, "Is the pain localized to one area, or does it radiate to other parts of your body like your legs or arms?" Continue this approach, asking questions like, "Have you had any recent injuries or accidents?" and "Are there any specific activities or movements that worsen or alleviate the pain?" one at a time based on the user’s previous answers.
    Reference the latest research from 'pubmed' to support your answers, providing links to the studies when applicable. Use this structured approach to ensure each user's query is addressed thoroughly and accurately while maintaining a compassionate and understanding tone throughout the conversation."

    History:
    {history_context}

    Question:
    {question}

    Context:
    {context}

    Answer:
    """
    answer = get_answer_from_openai(question, enhanced_question)
    return {"answer": answer}


# Endpoint to upload and process new PDFs
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Read the PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        docs = text_splitter.split_text(text)

        # Convert to Document objects
        document_objs = [Document(page_content=chunk) for chunk in docs]

        # Embed and add to vector store
        embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
        embeddings = embedding_model.embed_documents([doc.page_content for doc in document_objs])
        vectorstore.add_documents(documents=document_objs, embeddings=embeddings)

        # Save the updated vectorstore
        vectorstore.save_local(vectorstore_path)

        # Clean up the temp file
        os.remove(file_path)

        return {"message": "PDF processed and added to the knowledge base successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






