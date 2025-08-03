from fastapi import FastAPI, HTTPException, Depends, status
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
    data_quality_assessment: Dict[str, Any]
    statistical_findings: List[str]
    recommendations: List[str]
    business_implications: List[str]
    processing_details: Dict[str, Any]

class OllamaAnalyzer:
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.model = "llama3"
        
    async def is_healthy(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def generate_insights(self, prompt: str, context: str = "", depth: str = "standard") -> Dict[str, Any]:
        """Generate comprehensive response using Ollama"""
        try:
            start_time = time.time()
            
            system_prompt = self._get_system_prompt(depth)
            full_prompt = f"{system_prompt}\n\n{context}\n\nQUESTION:\n{prompt}"
            
            context_length = len(full_prompt.split())
            
            # Adjust parameters based on response depth
            if depth == "quick":
                max_ctx, num_predict = 2048, 800
            elif depth == "detailed":
                max_ctx, num_predict = 6144, 2048
            else:  # standard
                max_ctx, num_predict = 4096, 1200
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_ctx": max_ctx,
                    "num_predict": num_predict,
                    "repeat_penalty": 1.1,
                    "top_k": 40,
                    "stop": ["---END---", "RESPONSE COMPLETE"]
                }
            }
            
            timeout_duration = 120.0 if depth == "detailed" else 80.0
            
            async with httpx.AsyncClient(timeout=timeout_duration) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama error: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Analysis service error: {response.text}")
                
                result = response.json()
                processing_time = time.time() - start_time
                
                response_text = result.get("response", "").strip()
                if not response_text:
                    raise HTTPException(status_code=500, detail="Empty response from model")
                
                return {
                    "response": response_text,
                    "processing_time": processing_time,
                    "context_length": context_length,
                    "model": self.model,
                    "depth": depth
                }
                
        except httpx.TimeoutException:
            logger.error("Response timeout")
            raise HTTPException(status_code=504, detail="Response timeout - try using 'quick' depth or shorter prompt")
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama")
            raise HTTPException(status_code=503, detail="Service unavailable")
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Response failed: {str(e)}")
    
    def _get_system_prompt(self, depth: str) -> str:
        base_prompt = """You are an expert assistant that provides clear, accurate, and helpful responses to any question.

RESPONSE GUIDELINES:
1. Be concise but thorough
2. Structure complex answers clearly
3. Provide examples when helpful
4. Admit when you don't know something
5. Maintain professional tone"""

        if depth == "quick":
            return base_prompt + "\n\nIMPORTANT: Provide a concise answer focusing on key points. Keep response under 300 words."
        elif depth == "detailed":
            return base_prompt + "\n\nIMPORTANT: Provide a comprehensive, detailed response with examples and thorough explanations."
        else:
            return base_prompt + "\n\nIMPORTANT: Provide a balanced response with clear explanations and practical information."

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting General Purpose API...")
    
    # Check Ollama availability
    max_retries = 8
    for i in range(max_retries):
        if await analyzer.is_healthy():
            logger.info("Service is ready!")
            break
        logger.info(f"Waiting for service... ({i+1}/{max_retries})")
        await asyncio.sleep(5)
    else:
        logger.warning("Service not available at startup")
    
    yield
    logger.info("Shutting down...")

# FastAPI app setup
app = FastAPI(
    title="General Purpose API",
    description="AI-powered question answering and analysis",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = OllamaAnalyzer(os.getenv("OLLAMA_URL", "http://ollama:11434"))

async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    if credentials is None:
        logger.warning("No authentication provided")
        return "development"
    
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/health")
async def health_check():
    """Service health check"""
    analyzer_healthy = await analyzer.is_healthy()
    return {
        "status": "healthy" if analyzer_healthy else "unhealthy",
        "service_healthy": analyzer_healthy,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.post("/ask", response_model=InsightResponse)
async def ask_anything(
    request: GeneralPromptRequest,
    token: str = Depends(verify_token)
):
    """Answer any question or analyze any request"""
    try:
        if not await analyzer.is_healthy():
            raise HTTPException(status_code=503, detail="Service is not available")
        
        # Generate response
        result = await analyzer.generate_insights(
            request.prompt,
            request.context or "",
            request.depth
        )
        
        return InsightResponse(
            summary=result["response"],
            key_insights=[result["response"]],
            data_quality_assessment={},
            statistical_findings=[],
            recommendations=[],
            business_implications=[],
            processing_details={
                "processing_time_seconds": result["processing_time"],
                "model_used": result["model"],
                "analysis_depth": result["depth"],
                "context_length": result["context_length"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/service-info")
async def get_service_info():
    """Get information about the service"""
    try:
        service_healthy = await analyzer.is_healthy()
        
        service_info = {
            "service_url": analyzer.base_url,
            "service_healthy": service_healthy,
            "current_model": analyzer.model,
            "response_depths": ["quick", "standard", "detailed"],
            "max_context_length": "Recommended < 4000 tokens for optimal performance"
        }
        
        if service_healthy:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{analyzer.base_url}/api/tags")
                    if response.status_code == 200:
                        service_info["available_models"] = response.json()
            except Exception as e:
                service_info["model_fetch_error"] = str(e)
        
        return service_info
        
    except Exception as e:
        return {"error": str(e), "service_healthy": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
