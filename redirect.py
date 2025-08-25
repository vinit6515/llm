from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask-csv")
async def ask_csv(data: dict):
    return {"message": "Received!", "data": data}
