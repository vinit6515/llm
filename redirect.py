from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
import httpx
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can specify a list of allowed origins if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# External API URL
EXTERNAL_API_URL = "http://34.151.241.78:8000/ask-csv"

# Function to interact with the remote server (the external API)
async def call_external_server(content: str, filename: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:  # Increase timeout to 5 minutes (300 seconds)
            files = {"file": (filename, content, "text/csv")}
            response = await client.post(EXTERNAL_API_URL, files=files)
            if response.status_code != 200:
                logger.error(f"Error from external API: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to call external API")
            return response.json()  # Assuming the server returns a JSON response
    except httpx.TimeoutException:
        logger.error("External server timeout")
        raise HTTPException(status_code=504, detail="External API timeout")
    except httpx.RequestError as e:
        logger.error(f"Request to external server failed: {e}")
        raise HTTPException(status_code=503, detail="External API unavailable")


# Endpoint to handle CSV requests and call external server
@app.post("/ask-csv")
async def ask_csv(file: UploadFile = File(...)) -> dict:
    # Read the file content
    content = (await file.read()).decode("utf-8")
    filename = file.filename  # Get the filename to send it to the external API
    
    # Call the external server to process the content
    result = await call_external_server(content, filename)
    
    # Return the result received from the external server
    return result

