from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.post("/ask-csv")
async def redirect_to_backend(request: Request):
    return RedirectResponse(url="http://34.95.157.211:8000/ask-csv", status_code=307)
