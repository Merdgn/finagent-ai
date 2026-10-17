
from fastapi import FastAPI

app = FastAPI(
    title="FinAgent AI",
    description="Secure Banking Conversational AI Assistant",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "project": "FinAgent AI",
        "message": "Welcome to FinAgent AI",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
