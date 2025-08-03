from fastapi import FastAPI, HTTPException, Depends, status
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
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class DataAnalysisRequest(BaseModel):
    data: str = Field(..., description="Dataset in CSV format or JSON string")
    data_type: Optional[str] = Field("general", description="Type: sales, customer, financial, marketing, inventory, general")
    analysis_focus: Optional[List[str]] = Field(default=[], description="Specific areas to focus on (trends, patterns, anomalies, correlations)")
    analysis_depth: Optional[str] = Field("standard", description="Analysis depth: quick, standard, detailed")

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
        self.model = "llama3.1:8b"
        
    async def is_healthy(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def generate_insights(self, prompt: str, dataset_context: str, depth: str = "standard") -> Dict[str, Any]:
        """Generate comprehensive data insights using Ollama"""
        try:
            start_time = time.time()
            
            system_prompt = self._get_analysis_prompt(depth)
            full_prompt = f"{system_prompt}\n\n{dataset_context}\n\nANALYSIS REQUEST:\n{prompt}"
            
            context_length = len(full_prompt.split())
            
            # Adjust parameters based on analysis depth
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
                    "stop": ["---END---", "ANALYSIS COMPLETE"]
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
                    raise HTTPException(status_code=500, detail="Empty response from analysis model")
                
                return {
                    "response": response_text,
                    "processing_time": processing_time,
                    "context_length": context_length,
                    "model": self.model,
                    "depth": depth
                }
                
        except httpx.TimeoutException:
            logger.error("Analysis timeout")
            raise HTTPException(status_code=504, detail="Analysis timeout - try using 'quick' analysis depth or reduce data size")
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama")
            raise HTTPException(status_code=503, detail="Analysis service unavailable")
        except Exception as e:
            logger.error(f"Analysis generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _get_analysis_prompt(self, depth: str) -> str:
        base_prompt = """You are an expert data analyst and business intelligence specialist. Analyze the provided dataset and deliver comprehensive, actionable insights.

ANALYSIS STRUCTURE:
1. EXECUTIVE SUMMARY: Brief overview of key findings
2. DATA QUALITY ASSESSMENT: Evaluate completeness, accuracy, consistency
3. KEY INSIGHTS: Most important patterns, trends, and discoveries
4. STATISTICAL FINDINGS: Relevant statistical measures and their business meaning
5. BUSINESS IMPLICATIONS: What these findings mean for decision-making
6. ACTIONABLE RECOMMENDATIONS: Specific next steps

FOCUS AREAS:
- Identify significant patterns and trends
- Detect anomalies or outliers
- Find correlations and relationships
- Assess data quality issues
- Provide business context for technical findings
- Suggest concrete actions based on insights"""

        if depth == "quick":
            return base_prompt + "\n\nIMPORTANT: Provide a concise analysis focusing on the top 3 insights and 3 recommendations. Keep response under 600 words."
        elif depth == "detailed":
            return base_prompt + "\n\nIMPORTANT: Provide a comprehensive, detailed analysis with extensive statistical insights, multiple business scenarios, and thorough recommendations. Include specific examples from the data."
        else:
            return base_prompt + "\n\nIMPORTANT: Provide a balanced analysis with clear insights and practical recommendations. Structure your response with clear sections."

class DataStringProcessor:
    @staticmethod
    def process_string_data(data_str: str) -> Dict[str, Any]:
        """Process data string (CSV format) and extract comprehensive information"""
        try:
            data_str = data_str.strip()
            if not data_str:
                raise ValueError("Empty dataset provided")
            
            # Attempt to parse as CSV
            try:
                df = pd.read_csv(io.StringIO(data_str))
            except pd.errors.EmptyDataError:
                raise ValueError("No data found")
            except pd.errors.ParserError as e:
                # Try alternative parsing approaches
                try:
                    # Try with different separators
                    for sep in [';', '\t', '|']:
                        try:
                            df = pd.read_csv(io.StringIO(data_str), sep=sep)
                            if len(df.columns) > 1:
                                break
                        except:
                            continue
                    else:
                        raise ValueError(f"Unable to parse data format: {str(e)}")
                except:
                    raise ValueError(f"Data parsing error: {str(e)}")
            
            if df.empty:
                raise ValueError("Dataset contains no data")
            
            # Core dataset information
            basic_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
            }
            
            # Data quality metrics
            quality_metrics = DataStringProcessor._assess_data_quality(df)
            basic_info.update(quality_metrics)
            
            # Statistical summary
            stats_summary = DataStringProcessor._generate_statistical_summary(df)
            basic_info.update(stats_summary)
            
            # Convert to structured format for analysis
            if len(df) <= 100:
                # For smaller datasets, include full data
                df_clean = df.where(pd.notnull(df), None)
                basic_info["full_dataset"] = df_clean.to_dict('records')
            else:
                # For larger datasets, include sample
                sample_df = df.head(20)
                sample_clean = sample_df.where(pd.notnull(sample_df), None)
                basic_info["data_sample"] = sample_clean.to_dict('records')
                basic_info["sample_note"] = f"Showing first 20 rows of {len(df)} total rows"
            
            return basic_info
            
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    
    @staticmethod
    def _assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_info = {}
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_data = missing_counts[missing_counts > 0]
        
        if len(missing_data) > 0:
            quality_info["missing_values"] = missing_data.to_dict()
            missing_percentages = (missing_data / len(df) * 100).round(2)
            quality_info["missing_percentages"] = missing_percentages.to_dict()
        else:
            quality_info["missing_values"] = {}
            quality_info["missing_percentages"] = {}
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        quality_info["duplicate_rows"] = int(duplicate_count)
        quality_info["duplicate_percentage"] = round(duplicate_count / len(df) * 100, 2) if len(df) > 0 else 0
        
        # Calculate overall quality score
        completeness = (1 - (missing_counts.sum() / (len(df) * len(df.columns)))) * 100 if len(df) > 0 else 0
        consistency = (1 - (duplicate_count / len(df))) * 100 if len(df) > 0 else 100
        
        quality_score = (completeness * 0.6 + consistency * 0.4)
        quality_info["overall_quality_score"] = round(max(0, min(100, quality_score)), 1)
        
        return quality_info
    
    @staticmethod
    def _generate_statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary for different data types"""
        summary = {}
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary["numeric_columns"] = {}
            for col in numeric_cols:
                if df[col].nunique() > 1:
                    col_stats = {
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "unique_values": int(df[col].nunique())
                    }
                    summary["numeric_columns"][col] = col_stats
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            summary["categorical_columns"] = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count > 0:
                    col_info = {
                        "unique_values": unique_count,
                        "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                    }
                    
                    # Include value counts for categorical data with reasonable unique counts
                    if unique_count <= 20:
                        value_counts = df[col].value_counts().head(10)
                        col_info["value_distribution"] = value_counts.to_dict()
                    
                    summary["categorical_columns"][col] = col_info
        
        return summary
    
    @staticmethod
    def format_for_analysis(data_info: Dict[str, Any]) -> str:
        """Format processed data information for LLM analysis"""
        try:
            formatted = f"""DATASET OVERVIEW:
• Total Rows: {data_info['total_rows']:,}
• Total Columns: {data_info['total_columns']}
• Memory Usage: {data_info['memory_usage_kb']} KB
• Data Quality Score: {data_info.get('overall_quality_score', 'N/A')}/100

COLUMN INFORMATION:
{', '.join(data_info['column_names'])}

DATA TYPES:
"""
            for col, dtype in data_info['data_types'].items():
                formatted += f"• {col}: {dtype}\n"
            
            # Data quality details
            if data_info.get('missing_values'):
                formatted += f"\nDATA QUALITY ISSUES:\n"
                for col, count in data_info['missing_values'].items():
                    pct = data_info['missing_percentages'].get(col, 0)
                    formatted += f"• {col}: {count} missing values ({pct}%)\n"
            
            if data_info.get('duplicate_rows', 0) > 0:
                formatted += f"• Duplicate rows: {data_info['duplicate_rows']} ({data_info.get('duplicate_percentage', 0)}%)\n"
            
            # Statistical summaries
            if data_info.get('numeric_columns'):
                formatted += f"\nNUMERIC COLUMNS ANALYSIS:\n"
                for col, stats in data_info['numeric_columns'].items():
                    formatted += f"• {col}: mean={stats.get('mean')}, range=[{stats.get('min')} to {stats.get('max')}], unique values={stats.get('unique_values')}\n"
            
            if data_info.get('categorical_columns'):
                formatted += f"\nCATEGORICAL COLUMNS ANALYSIS:\n"
                for col, info in data_info['categorical_columns'].items():
                    formatted += f"• {col}: {info.get('unique_values')} unique values"
                    if info.get('most_frequent'):
                        formatted += f", most frequent: '{info['most_frequent']}'"
                    formatted += "\n"
            
            # Include actual data
            if data_info.get('full_dataset'):
                formatted += f"\nFULL DATASET (JSON format):\n"
                formatted += json.dumps(data_info['full_dataset'], indent=2, default=str)
            elif data_info.get('data_sample'):
                formatted += f"\nDATA SAMPLE ({data_info.get('sample_note', 'Sample data')}):\n"
                formatted += json.dumps(data_info['data_sample'], indent=2, default=str)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting data: {e}")
            return f"Dataset with {data_info.get('total_rows', 'unknown')} rows and {data_info.get('total_columns', 'unknown')} columns. Error in detailed formatting: {str(e)}"

# Initialize components
analyzer = OllamaAnalyzer(os.getenv("OLLAMA_URL", "http://ollama:11434"))
processor = DataStringProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Data Analysis API...")
    
    # Check Ollama availability
    max_retries = 8
    for i in range(max_retries):
        if await analyzer.is_healthy():
            logger.info("Analysis service is ready!")
            break
        logger.info(f"Waiting for analysis service... ({i+1}/{max_retries})")
        await asyncio.sleep(5)
    else:
        logger.warning("Analysis service not available at startup")
    
    yield
    logger.info("Shutting down...")

# FastAPI app setup
app = FastAPI(
    title="Data Insights API",
    description="AI-powered data analysis for string-based datasets",
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
        "analyzer_service": analyzer_healthy,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.post("/analyze-data", response_model=InsightResponse)
async def analyze_data_string(
    request: DataAnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyze data provided as string and generate comprehensive insights"""
    try:
        # Validate input
        if not request.data or request.data.strip() == "":
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Check service availability
        if not await analyzer.is_healthy():
            raise HTTPException(status_code=503, detail="Analysis service is not available")
        
        # Process the data string
        data_info = processor.process_string_data(request.data)
        formatted_data = processor.format_for_analysis(data_info)
        
        # Create comprehensive analysis prompt
        focus_areas = ", ".join(request.analysis_focus) if request.analysis_focus else "comprehensive business insights"
        
        analysis_prompt = f"""
Analyze this {request.data_type} dataset with focus on: {focus_areas}

Please provide:
1. Executive summary of key findings
2. Data quality assessment and recommendations
3. Statistical insights with business implications
4. Pattern recognition and trend analysis
5. Actionable recommendations for stakeholders

Analysis depth: {request.analysis_depth}
"""
        
        # Generate insights
        result = await analyzer.generate_insights(
            analysis_prompt, 
            formatted_data, 
            request.analysis_depth
        )
        
        # Parse structured response
        insights_text = result["response"]
        parsed_insights = DataStringProcessor._parse_insights(insights_text, data_info)
        
        return InsightResponse(
            summary=parsed_insights["summary"],
            key_insights=parsed_insights["key_insights"],
            data_quality_assessment={
                "overall_score": data_info.get('overall_quality_score', 0),
                "missing_data": data_info.get('missing_values', {}),
                "duplicates": data_info.get('duplicate_rows', 0),
                "recommendations": parsed_insights["quality_recommendations"]
            },
            statistical_findings=parsed_insights["statistical_findings"],
            recommendations=parsed_insights["recommendations"],
            business_implications=parsed_insights["business_implications"],
            processing_details={
                "processing_time_seconds": result["processing_time"],
                "model_used": result["model"],
                "analysis_depth": result["depth"],
                "context_length": result["context_length"],
                "dataset_size": f"{data_info['total_rows']} rows × {data_info['total_columns']} columns"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @staticmethod
    def _parse_insights(insights_text: str, data_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Parse the insights text into structured components"""
        lines = insights_text.split('\n')
        
        parsed = {
            "summary": "",
            "key_insights": [],
            "statistical_findings": [],
            "recommendations": [],
            "business_implications": [],
            "quality_recommendations": []
        }
        
        current_section = None
        summary_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Detect sections
            if any(keyword in line_lower for keyword in ['executive summary', 'summary', 'overview']):
                current_section = 'summary'
                continue
            elif any(keyword in line_lower for keyword in ['key insights', 'key findings', 'main insights']):
                current_section = 'key_insights'
                continue
            elif any(keyword in line_lower for keyword in ['statistical', 'statistics', 'statistical findings']):
                current_section = 'statistical_findings'
                continue
            elif any(keyword in line_lower for keyword in ['recommendations', 'recommendation', 'suggestions']):
                current_section = 'recommendations'
                continue
            elif any(keyword in line_lower for keyword in ['business implications', 'business impact', 'implications']):
                current_section = 'business_implications'
                continue
            elif any(keyword in line_lower for keyword in ['data quality', 'quality']):
                current_section = 'quality_recommendations'
                continue
            
            # Process content based on current section
            if current_section == 'summary':
                if not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    summary_lines.append(line)
            elif current_section and line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line and len(clean_line) > 15:
                    parsed[current_section].append(clean_line)
        
        # Create summary from collected lines or generate default
        parsed["summary"] = ' '.join(summary_lines) if summary_lines else f"Analysis of {data_info['total_rows']} records across {data_info['total_columns']} dimensions with {data_info.get('overall_quality_score', 0)}/100 quality score."
        
        # Ensure minimum content
        if not parsed["key_insights"]:
            parsed["key_insights"] = [
                f"Dataset contains {data_info['total_rows']} records with {data_info['total_columns']} variables",
                f"Data quality score: {data_info.get('overall_quality_score', 0)}/100",
                f"Missing data affects {len(data_info.get('missing_values', {}))} columns" if data_info.get('missing_values') else "Complete dataset with no missing values"
            ]
        
        if not parsed["recommendations"]:
            parsed["recommendations"] = [
                "Review data collection processes to ensure consistency",
                "Implement data validation rules for improved quality",
                "Consider additional analysis based on business objectives"
            ]
        
        return parsed

# Add the static method to the DataStringProcessor class
DataStringProcessor._parse_insights = lambda insights_text, data_info: DataStringProcessor._parse_insights_static(insights_text, data_info)

@app.get("/service-info")
async def get_service_info():
    """Get information about the analysis service"""
    try:
        service_healthy = await analyzer.is_healthy()
        
        service_info = {
            "service_url": analyzer.base_url,
            "service_healthy": service_healthy,
            "current_model": analyzer.model,
            "supported_data_types": ["sales", "customer", "financial", "marketing", "inventory", "general"],
            "analysis_depths": ["quick", "standard", "detailed"],
            "max_dataset_size": "Recommended < 10MB for optimal performance"
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
