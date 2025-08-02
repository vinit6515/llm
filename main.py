from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import httpx
import pandas as pd
import numpy as np
import json
import io
import logging
import time
import os
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class DataRequest(BaseModel):
    data: str = Field(..., description="Dataset in CSV format or JSON")
    data_type: Optional[str] = Field("general", description="Type: sales, customer, financial, general")
    analysis_focus: Optional[List[str]] = Field(default=[], description="Specific areas to focus on")
    include_visualizations: Optional[bool] = Field(False, description="Include visualization recommendations")

class FileAnalysisRequest(BaseModel):
    data_type: Optional[str] = Field("general", description="Type of data analysis")
    analysis_focus: Optional[List[str]] = Field(default=[], description="Specific areas to focus on")

class AnalysisResponse(BaseModel):
    insights: str
    data_quality_score: float
    key_findings: List[str]
    recommendations: List[str]
    processing_time: float
    model_used: str
    context_length: int

class OllamaClient:
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.model = "llama3.1:8b"  # Default model
        
    async def is_healthy(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def generate_analysis(self, prompt: str, context_data: str) -> Dict[str, Any]:
        """Generate analysis using Ollama"""
        try:
            start_time = time.time()
            
            # Construct a more concise prompt to avoid timeouts
            system_prompt = self._get_concise_system_prompt()
            full_prompt = f"{system_prompt}\n\nDATA:\n{context_data}\n\nTASK: {prompt}"
            
            # Calculate context length
            context_length = len(full_prompt.split())
            
            # Use smaller context window and shorter responses for faster processing
            max_ctx = min(4096, max(1024, context_length + 500))
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower temperature for faster, more focused responses
                    "top_p": 0.8,
                    "num_ctx": max_ctx,
                    "repeat_penalty": 1.05,
                    "top_k": 20,  # Reduced for faster processing
                    "num_predict": 1024,  # Shorter response to avoid timeouts
                    "stop": ["---", "END"]  # Stop sequences to limit response length
                }
            }
            
            # Much shorter timeout for small datasets
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama error response: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Ollama request failed: {response.text}")
                
                result = response.json()
                processing_time = time.time() - start_time
                
                # Check if response is empty or contains error
                response_text = result.get("response", "").strip()
                if not response_text:
                    raise HTTPException(status_code=500, detail="Empty response from model")
                
                return {
                    "response": response_text,
                    "processing_time": processing_time,
                    "context_length": context_length,
                    "model": self.model
                }
                
        except httpx.TimeoutException:
            logger.error("Ollama request timeout")
            raise HTTPException(status_code=504, detail="Analysis timeout - try using a smaller model or reducing data size")
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama service")
            raise HTTPException(status_code=503, detail="Analysis service unavailable - check if Ollama is running")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        return """You are an expert data analyst with extensive experience in statistical analysis, data quality assessment, and business intelligence. Your task is to analyze datasets and provide comprehensive, actionable insights.

ANALYSIS FRAMEWORK:
1. DATA OVERVIEW: Summarize the dataset structure, size, and key characteristics
2. KEY INSIGHTS: Identify the most important patterns, trends, and findings
3. DATA QUALITY ASSESSMENT: Evaluate completeness, accuracy, consistency, and validity
4. STATISTICAL ANALYSIS: Provide relevant statistical measures and their interpretations
5. BUSINESS IMPLICATIONS: Explain what the findings mean for business decisions
6. RECOMMENDATIONS: Suggest specific, actionable next steps

QUALITY SCORING:
Rate data quality from 0-100 based on:
- Completeness (missing values)
- Consistency (data formats, duplicates)
- Accuracy (outliers, invalid values)
- Timeliness (data freshness)
- Validity (business rule compliance)

OUTPUT FORMAT:
Structure your response clearly with sections. Be thorough but concise, focusing on actionable insights rather than technical jargon. Always provide at least 3 key findings and 3 recommendations."""
    
    def _get_concise_system_prompt(self) -> str:
        return """You are a data analyst. Analyze the provided dataset and give:

1. DATA OVERVIEW: Dataset size, structure, columns
2. KEY FINDINGS: Top 3 important patterns or insights  
3. DATA QUALITY: Rate 0-100, note any issues
4. RECOMMENDATIONS: Top 3 actionable suggestions

Be concise and focus on the most important insights. Keep response under 500 words."""

class DataProcessor:
    @staticmethod
    def process_csv_data(data_str: str) -> Dict[str, Any]:
        """Process CSV data and extract key information, converting to JSON format"""
        try:
            # Clean the data string
            data_str = data_str.strip()
            if not data_str:
                raise ValueError("Empty dataset provided")
            
            # Read CSV data with error handling
            try:
                df = pd.read_csv(io.StringIO(data_str))
            except pd.errors.EmptyDataError:
                raise ValueError("No data found in CSV")
            except pd.errors.ParserError as e:
                raise ValueError(f"CSV parsing error: {str(e)}")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Convert entire dataset to JSON format for better processing
            # Replace NaN values with None for JSON serialization
            df_clean = df.where(pd.notnull(df), None)
            json_data = df_clean.to_dict('records')
            
            # Basic info
            info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "full_dataset_json": json_data,  # Include full dataset as JSON
            }
            
            # Simplified missing values analysis
            missing_data = df.isnull().sum()
            info["missing_values"] = missing_data[missing_data > 0].to_dict()
            
            # Calculate missing percentages safely
            if len(df) > 0:
                missing_pct = (missing_data / len(df) * 100).round(2)
                info["missing_percentage"] = missing_pct[missing_pct > 0].to_dict()
            else:
                info["missing_percentage"] = {}
            
            # Quick numeric summary (only for small datasets)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(df) <= 100:  # Only for small datasets
                try:
                    info["numeric_summary"] = {}
                    for col in numeric_cols:
                        if df[col].nunique() > 1:  # Only if there's variation
                            info["numeric_summary"][col] = {
                                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                                "unique_values": int(df[col].nunique())
                            }
                except Exception as e:
                    logger.warning(f"Error in numeric summary: {e}")
                    info["numeric_summary"] = {}
            
            # Quick categorical summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                info["categorical_summary"] = {}
                for col in categorical_cols:
                    try:
                        unique_count = df[col].nunique()
                        if unique_count <= 10 and unique_count > 0:
                            value_counts = df[col].value_counts().head(3)  # Only top 3
                            info["categorical_summary"][col] = value_counts.to_dict()
                    except Exception as e:
                        logger.warning(f"Error processing categorical column {col}: {e}")
                        continue
            
            # Data quality metrics
            quality_score = DataProcessor._calculate_quality_score(df)
            info["quality_score"] = quality_score
            
            return info
            
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV data: {str(e)}")
    
    @staticmethod
    def _calculate_quality_score(df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Completeness score (100 - missing percentage)
            total_cells = len(df) * len(df.columns)
            if total_cells > 0:
                missing_cells = df.isnull().sum().sum()
                missing_pct = (missing_cells / total_cells) * 100
                completeness = max(0, 100 - missing_pct)
                scores.append(completeness)
            else:
                scores.append(0)
            
            # Consistency score (based on duplicates)
            if len(df) > 0:
                duplicate_count = df.duplicated().sum()
                duplicate_pct = (duplicate_count / len(df)) * 100
                consistency = max(0, 100 - duplicate_pct * 2)  # Penalize duplicates more
                scores.append(consistency)
            else:
                scores.append(100)
            
            # Validity score (based on data types and outliers)
            validity = 100  # Start with perfect score
            
            # Check for outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    if df[col].nunique() > 1:  # Only check if there's variation
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:  # Avoid division by zero
                            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
                            outlier_pct = len(outliers) / len(df) * 100
                            validity -= min(outlier_pct, 20)  # Cap penalty at 20 points
                except Exception as e:
                    logger.warning(f"Error calculating outliers for column {col}: {e}")
                    continue
            
            scores.append(max(0, validity))
            
            # Return average score
            return round(np.mean(scores), 1) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 50.0  # Default middle score if calculation fails
    
    @staticmethod
    def format_data_for_analysis(data_info: Dict[str, Any]) -> str:
        """Format processed data info for LLM analysis using JSON format"""
        try:
            # For small datasets (like 13 rows), include the full JSON data
            if data_info['rows'] <= 50:
                formatted = f"""DATASET OVERVIEW:
- Rows: {data_info['rows']}
- Columns: {data_info['columns']}
- Column Names: {', '.join(data_info['column_names'])}

DATA QUALITY SCORE: {data_info.get('quality_score', 'N/A')}/100

FULL DATASET (JSON format):
{json.dumps(data_info['full_dataset_json'], indent=2, default=str)}"""
            else:
                # For larger datasets, use summary approach
                formatted = f"""DATASET OVERVIEW:
- Rows: {data_info['rows']:,}
- Columns: {data_info['columns']}
- Column Names: {', '.join(data_info['column_names'])}
- Memory Usage: {data_info['memory_usage'] / 1024:.1f} KB

DATA QUALITY SCORE: {data_info.get('quality_score', 'N/A')}/100"""

                # Add data types information
                if data_info.get('data_types'):
                    formatted += f"\n\nDATA TYPES:\n"
                    for col, dtype in data_info['data_types'].items():
                        formatted += f"- {col}: {dtype}\n"

                # Add sample of the JSON data
                if data_info.get('full_dataset_json'):
                    sample_size = min(10, len(data_info['full_dataset_json']))
                    formatted += f"\nSAMPLE DATA (first {sample_size} rows in JSON):\n"
                    formatted += json.dumps(data_info['full_dataset_json'][:sample_size], indent=2, default=str)

            # Add missing values info if present
            if data_info.get('missing_values'):
                formatted += f"\n\nMISSING VALUES:\n"
                for col, count in data_info['missing_values'].items():
                    pct = data_info['missing_percentage'].get(col, 0)
                    formatted += f"- {col}: {count} ({pct}%)\n"

            # Add quick numeric summary for small datasets
            if data_info.get('numeric_summary'):
                formatted += f"\nNUMERIC COLUMNS SUMMARY:\n"
                for col, stats in data_info['numeric_summary'].items():
                    formatted += f"- {col}: mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}\n"

            # Add categorical summary
            if data_info.get('categorical_summary'):
                formatted += f"\nCATEGORICAL COLUMNS (top values):\n"
                for col, values in data_info['categorical_summary'].items():
                    top_values = list(values.keys())[:2]  # Only show top 2
                    formatted += f"- {col}: {', '.join(map(str, top_values))}\n"

            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting data for analysis: {e}")
            # Fallback to minimal format
            return f"""DATASET OVERVIEW:
- Rows: {data_info.get('rows', 'Unknown')}
- Columns: {data_info.get('columns', 'Unknown')}
- Quality Score: {data_info.get('quality_score', 'Unknown')}/100

DATA (JSON format):
{json.dumps(data_info.get('full_dataset_json', [])[:5], indent=2, default=str)}

Error formatting details: {str(e)}"""

# Initialize global objects
ollama_client = OllamaClient(os.getenv("OLLAMA_URL", "http://ollama:11434"))
data_processor = DataProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Data Analysis API...")
    
    # Wait for Ollama to be ready
    max_retries = 10  # Reduced retries for faster startup
    for i in range(max_retries):
        if await ollama_client.is_healthy():
            logger.info("Ollama is ready!")
            break
        logger.info(f"Waiting for Ollama... ({i+1}/{max_retries})")
        await asyncio.sleep(5)  # Reduced wait time
    else:
        logger.warning("Ollama not available at startup - will retry on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Data Analysis API",
    description="AI-powered data analysis and insights generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make authentication optional for testing
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    # For development, make auth optional
    if credentials is None:
        logger.warning("No authentication provided - this should be fixed in production")
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
    """Health check endpoint"""
    ollama_healthy = await ollama_client.is_healthy()
    return {
        "status": "healthy" if ollama_healthy else "unhealthy",
        "ollama": ollama_healthy,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(
    request: DataRequest,
    token: str = Depends(verify_token)
):
    """Analyze provided dataset"""
    try:
        # Validate input
        if not request.data or request.data.strip() == "":
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Check if Ollama is available
        if not await ollama_client.is_healthy():
            raise HTTPException(status_code=503, detail="Analysis service is not available")
        
        # Process the data
        data_info = data_processor.process_csv_data(request.data)
        formatted_data = data_processor.format_data_for_analysis(data_info)
        
        # Create analysis prompt
        focus_areas = ", ".join(request.analysis_focus) if request.analysis_focus else "general insights"
        prompt = f"""Please analyze this {request.data_type} dataset focusing on {focus_areas}. 
        Provide comprehensive insights, identify any data quality issues, and suggest actionable recommendations.
        
        Please structure your response with clear sections and provide at least 3 key findings and 3 recommendations."""
        
        # Generate analysis
        result = await ollama_client.generate_analysis(prompt, formatted_data)
        
        # Parse the response to extract structured information
        insights = result["response"]
        
        # Extract key findings and recommendations with better parsing
        key_findings = []
        recommendations = []
        
        lines = insights.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Detect sections
            if any(keyword in line_lower for keyword in ['key insights', 'findings', 'key finding']):
                current_section = 'findings'
                continue
            elif any(keyword in line_lower for keyword in ['recommendations', 'recommendation']):
                current_section = 'recommendations'
                continue
            elif any(keyword in line_lower for keyword in ['overview', 'summary', 'analysis']):
                current_section = None
                continue
            
            # Extract bullet points or numbered items
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line and len(clean_line) > 10:  # Only meaningful findings
                    if current_section == 'findings':
                        key_findings.append(clean_line)
                    elif current_section == 'recommendations':
                        recommendations.append(clean_line)
        
        # If no structured findings found, create some based on data
        if not key_findings:
            key_findings = [
                f"Dataset contains {data_info['rows']} rows and {data_info['columns']} columns",
                f"Data quality score: {data_info.get('quality_score', 0)}/100",
                f"Missing values detected in {len(data_info.get('missing_values', {}))} columns" if data_info.get('missing_values') else "No missing values detected"
            ]
        
        if not recommendations:
            recommendations = [
                "Review data quality and handle missing values if present",
                "Validate data types and correct any inconsistencies",
                "Consider additional data collection for better insights"
            ]
        
        return AnalysisResponse(
            insights=insights,
            data_quality_score=data_info.get('quality_score', 0.0),
            key_findings=key_findings[:5],  # Top 5 findings
            recommendations=recommendations[:5],  # Top 5 recommendations
            processing_time=result["processing_time"],
            model_used=result["model"],
            context_length=result["context_length"]
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    data_type: str = "general",
    token: str = Depends(verify_token)
):
    """Analyze uploaded CSV file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content with encoding handling
        content = await file.read()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                data_str = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Unable to decode file - unsupported encoding")
        
        # Create request object
        request = DataRequest(
            data=data_str,
            data_type=data_type
        )
        
        # Use existing analyze_data function
        return await analyze_data(request, token)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"File analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@app.post("/test-quick-analysis")
async def test_quick_analysis(
    request: DataRequest,
    token: str = Depends(verify_token)
):
    """Quick test analysis with minimal processing to debug timeouts"""
    try:
        # Process the data
        data_info = data_processor.process_csv_data(request.data)
        
        # Create a very simple prompt for testing
        simple_prompt = f"Briefly analyze this dataset with {data_info['rows']} rows and {data_info['columns']} columns. List 2 key observations."
        
        # Use only basic info to avoid context issues
        basic_data = f"Dataset: {data_info['rows']} rows, {data_info['columns']} columns\nColumns: {', '.join(data_info['column_names'])}\nQuality Score: {data_info.get('quality_score', 0)}"
        
        # Test with minimal settings
        payload = {
            "model": ollama_client.model,
            "prompt": f"{simple_prompt}\n\nData: {basic_data}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 1024,
                "num_predict": 200
            }
        }
        
        start_time = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_client.base_url}/api/generate",
                json=payload
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                return {"error": f"Ollama failed: {response.text}", "status_code": response.status_code}
            
            result = response.json()
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "processing_time": processing_time,
                "data_info": {
                    "rows": data_info['rows'],
                    "columns": data_info['columns'],
                    "quality_score": data_info.get('quality_score', 0)
                }
            }
        
    except Exception as e:
        logger.error(f"Test analysis error: {e}")
        return {"error": str(e), "success": False}

@app.get("/debug-info")
async def debug_info():
    """Get debug information about the service"""
    try:
        ollama_healthy = await ollama_client.is_healthy()
        
        # Try to get model info
        model_info = None
        if ollama_healthy:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{ollama_client.base_url}/api/tags")
                    if response.status_code == 200:
                        model_info = response.json()
            except Exception as e:
                model_info = {"error": str(e)}
        
        return {
            "ollama_url": ollama_client.base_url,
            "ollama_healthy": ollama_healthy,
            "current_model": ollama_client.model,
            "available_models": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/models")
async def get_available_models(token: str = Depends(verify_token)):
    """Get list of available models"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{ollama_client.base_url}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=500, detail="Failed to fetch models")
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch-model")
async def switch_model(model_name: str, token: str = Depends(verify_token)):
    """Switch to a different model"""
    try:
        ollama_client.model = model_name
        return {"message": f"Switched to model: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
