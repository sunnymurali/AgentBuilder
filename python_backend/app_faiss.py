from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import os
import io
import json
import traceback
from datetime import datetime

# Import document processing utilities
from python_backend.document_processor import process_document, chunk_text

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class AgentBase(BaseModel):
    name: str
    systemPrompt: str
    model: str

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    systemPrompt: Optional[str] = None
    model: Optional[str] = None

class AgentCreateResponse(AgentBase):
    id: int
    createdAt: str
    updatedAt: str

class MessageBase(BaseModel):
    agentId: int
    role: str
    content: str

class MessageResponse(MessageBase):
    id: int
    timestamp: str

class MessagePair(BaseModel):
    userMessage: MessageResponse
    assistantMessage: MessageResponse

class ErrorResponse(BaseModel):
    message: str
    error: Optional[str] = None

# Initialize storage with FAISS vector store
try:
    from python_backend.faiss_storage import FAISSVectorStorage
    # Use FAISS storage for agents, messages, and documents
    storage = FAISSVectorStorage()
    using_fallback = False
    print("FAISS vector storage initialized successfully")
    
    # Import RAG processing with FAISS
    from python_backend.faiss_rag import process_query
    
    # Document manager is the same as the storage in this implementation
    document_manager = storage
    
except ImportError as e:
    print(f"FAISS vector storage initialization failed: {e}")
    # Fall back to memory storage if FAISS fails
    from python_backend.fallback_storage import MemoryStorage
    storage = MemoryStorage()
    using_fallback = True
    
    # Simplified query processing for fallback mode
    def process_query(query, messages, agent_id=None):
        # Basic response without RAG
        return {"response": "Document retrieval is not available in fallback mode."}
    
    document_manager = storage
    print("Using fallback storage (MemoryStorage) as FAISS is not available")

# Create default agents if no agents exist yet
agents = storage.get_agents()
if not agents:
    storage.create_agent({
        "name": "General Assistant",
        "systemPrompt": "You are a helpful assistant that provides accurate and concise information.",
        "model": "gpt-4o"
    })

# API routes
@app.get("/api/agents", response_model=List[AgentCreateResponse])
async def get_agents():
    """Get all agents"""
    try:
        return storage.get_agents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}", response_model=AgentCreateResponse)
async def get_agent(agent_id: int):
    """Get a specific agent by ID"""
    agent = storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/api/agents", response_model=AgentCreateResponse)
async def create_agent(agent: AgentBase):
    """Create a new agent"""
    try:
        new_agent = storage.create_agent(agent.dict())
        return new_agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/agents/{agent_id}", response_model=AgentCreateResponse)
async def update_agent(agent_id: int, agent_update: AgentUpdate):
    """Update an existing agent"""
    updated_agent = storage.update_agent(agent_id, agent_update.dict(exclude_unset=True))
    if not updated_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return updated_agent

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int):
    """Delete an agent and all associated data"""
    success = storage.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

@app.get("/api/agents/{agent_id}/messages", response_model=List[MessageResponse])
async def get_agent_messages(agent_id: int):
    """Get all messages for a specific agent"""
    return storage.get_agent_messages(agent_id)

@app.post("/api/agents/{agent_id}/messages", response_model=MessagePair)
async def create_message(agent_id: int, message: MessageBase):
    """Create a new message and generate a response"""
    try:
        # Get the agent to verify it exists
        agent = storage.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create the user message
        user_message = storage.create_message({
            "agentId": agent_id,
            "role": "user",
            "content": message.content
        })
        
        # Get all messages for context
        all_messages = storage.get_agent_messages(agent_id)
        
        # Generate a response using the RAG process
        rag_response = process_query(message.content, all_messages, agent_id)
        
        # Create the assistant message with the generated response
        assistant_message = storage.create_message({
            "agentId": agent_id,
            "role": "assistant",
            "content": rag_response["response"]
        })
        
        # Return both messages
        return {"userMessage": user_message, "assistantMessage": assistant_message}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}/documents")
async def get_agent_documents(agent_id: int):
    """Get all documents for a specific agent"""
    return document_manager.get_agent_documents(agent_id)

@app.post("/api/agents/{agent_id}/documents")
async def upload_document(
    agent_id: int, 
    file: UploadFile = File(...),
    description: str = Form(None)
):
    """Upload and process a document for a specific agent"""
    try:
        # Read the file content
        content = await file.read()
        
        # Process the document
        document_info = process_document(
            io.BytesIO(content),
            file.filename,
            agent_id,
            description
        )
        
        # Extract text chunks
        text = document_info.get("text", "")
        text_chunks = chunk_text(text)
        
        # Add the document to storage
        document = document_manager.add_document(document_info, text_chunks)
        
        # Remove the full text to keep response size smaller
        if "text" in document:
            del document["text"]
        
        return document
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID"""
    document = document_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    success = document_manager.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@app.post("/api/rag")
async def generate_rag_response(agent_id: int, request: Request):
    """Generate a response using RAG"""
    try:
        # Parse the request body
        body = await request.json()
        query = body.get("query", "")
        messages = body.get("messages", [])
        
        # Generate a response using the RAG process
        response = process_query(query, messages, agent_id)
        
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Finance AI Assistant API with FAISS"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}