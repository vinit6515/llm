from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import httpx
import logging
import time
import os
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class GeneralPromptRequest(BaseModel):
    prompt: str = Field(..., description="Your question or analysis request")
    context: Optional[str] = Field(None, description="Additional context for the request")
    depth: Optional[str] = Field("standard", description="Response depth: quick, standard, detailed")

class InsightResponse(BaseModel):
    summary: str
    key_insights: List[str]
    processing_details: Dict[str, Any]

class OllamaAnalyzer:
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.model = "mistral-7b-instruct"  # Updated model
    
    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def generate_insights(self, prompt: str, context: str = "", depth: str = "standard") -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Optimized system prompt for large data
            system_prompt = """You are an expert data analyst processing large datasets. 
            Provide concise insights focusing on key patterns, anomalies, and trends.
            Structure response with bullet points for clarity."""
            
            full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{prompt}"
            
            # Configuration for mistral-7b-instruct-64k
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower for analytical tasks
                    "num_ctx": 65536,    # Utilize full 64K context
                    "num_predict": 4096,  # Max tokens to generate
                    "top_k": 50,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Dynamic timeout based on depth
            timeout = httpx.Timeout(
                connect=30.0,
                read=300.0 if depth == "detailed" else 180.0
            )
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama error: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Analysis error: {response.text}")
                
                result = response.json()
                return {
                    "response": result.get("response", "").strip(),
                    "processing_time": time.time() - start_time,
                    "model": self.model,
                    "context_length": len(full_prompt.split())
                }
                
        except httpx.TimeoutException:
            logger.error("Processing timeout")
            raise HTTPException(
                status_code=504,
                detail="Try reducing context size or using 'quick' mode"
            )
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API with mistral-7b-instruct-64k...")
    analyzer = OllamaAnalyzer()
    
    # Verify model availability
    if not await analyzer.is_healthy():
        logger.error("Ollama service unavailable")
        raise RuntimeError("Ollama not running")
    
    # Check if model exists
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{analyzer.base_url}/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        if analyzer.model not in models:
            logger.info(f"Downloading {analyzer.model}...")
            await client.post(f"{analyzer.base_url}/api/pull", json={"name": analyzer.model})
    
    yield
    logger.info("Shutdown complete")

app = FastAPI(
    title="Large Data Analysis API",
    description="Process massive datasets with mistral-7b-instruct-64k",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=InsightResponse)
async def analyze_data(
    request: GeneralPromptRequest,
    token: str = Depends(security)
):
    """Endpoint optimized for large data processing"""
    analyzer = OllamaAnalyzer()
    
    try:
        result = await analyzer.generate_insights(
            prompt=request.prompt,
            context=request.context or "",
            depth=request.depth
        )
        
        return InsightResponse(
            summary=result["response"],
            key_insights=[result["response"]],  # Or parse into bullet points
            processing_details={
                "time_sec": round(result["processing_time"], 2),
                "model": result["model"],
                "context_tokens": result["context_length"]
            }
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    analyzer = OllamaAnalyzer()
    return {
        "status": "healthy" if await analyzer.is_healthy() else "unhealthy",
        "model": analyzer.model,
        "max_context": "64K tokens"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300  # Important for long requests
    )
