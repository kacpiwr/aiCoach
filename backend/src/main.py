from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from src.analysis.analyze_user_shot import UserShotAnalyzer

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "shot_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_shot(file: UploadFile = File(...), user: str = Form(...)):
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{user}_{file.filename}")
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    try:
        # Analyze the shot
        analyzer = UserShotAnalyzer()
        results_path = analyzer.analyze_shot(file_path, user, method="ml")
        
        # Read and return the results
        with open(results_path, "r") as f:
            results = f.read()
        
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Shot Analysis API is running"} 