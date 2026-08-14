import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.feature.classifier import classify_email

app = FastAPI(
    title="AI Model Classification Microservice",
    description="FastAPI microservice wrapping PyTorch GPT-2 model for email classification",
    version="1.0.0"
)

# Enable CORS for local cross-origin communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Customer support email text to classify")


class PredictResponse(BaseModel):
    category: str = Field(..., description="Predicted email category")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fastapi-llm-classifier"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
    
    try:
        # Run inference using the existing PyTorch model pipeline
        result = classify_email(request.text, prompt_version="v1")
        category_str = result.category.value if hasattr(result.category, 'value') else str(result.category)
        return PredictResponse(category=category_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
