from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
import json
import logging
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
USERS_SHOTS_DIR = BASE_DIR / "data" / "raw" / "users_shots"
RESULTS_DIR = BASE_DIR / "data" / "results" / "analysis" / "analysis"
MODELS_DIR = BASE_DIR / "data" / "models"
REFERENCE_DIR = BASE_DIR / "data" / "reference" / "reference_shots"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create necessary directories
USERS_SHOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for the analyzer
os.environ["MODELS_DIR"] = str(MODELS_DIR)
os.environ["REFERENCE_DIR"] = str(REFERENCE_DIR)
os.environ["PROCESSED_DIR"] = str(PROCESSED_DIR)
os.environ["RESULTS_DIR"] = str(RESULTS_DIR)
os.environ["SHOT_DATA_DIR"] = str(BASE_DIR / "data" / "shot_data")
os.environ["REFERENCE_SHOTS_DIR"] = str(REFERENCE_DIR)
os.environ["USER_SHOTS_DIR"] = str(USERS_SHOTS_DIR)

# Log the paths for debugging
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Users shots directory: {USERS_SHOTS_DIR}")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Reference directory: {REFERENCE_DIR}")
logger.info(f"Processed directory: {PROCESSED_DIR}")
logger.info(f"Shot data directory: {os.environ['SHOT_DATA_DIR']}")
logger.info(f"Reference shots directory: {os.environ['REFERENCE_SHOTS_DIR']}")
logger.info(f"User shots directory: {os.environ['USER_SHOTS_DIR']}")

app = FastAPI(
    title="Shot Analysis API",
    description="API for analyzing basketball shots using computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_recommendations(text):
    """Parse the recommendations text into a structured format"""
    sections = {
        "overall": {},
        "stages": {},
        "detailed_recommendations": [],
        "practice_drills": [],
        "progress_tracking": []
    }
    
    # Extract overall similarity
    overall_match = re.search(r"Overall Similarity to Curry's Form: ([\d.]+)%", text)
    if overall_match:
        sections["overall"]["similarity"] = float(overall_match.group(1))
    
    # Extract stage-by-stage analysis
    stage_pattern = r"(\w+) STAGE:\nSimilarity: ([\d.]+)%\nStrengths:\n(.*?)(?=\n\n\w+ STAGE:|$)"
    stage_matches = re.finditer(stage_pattern, text, re.DOTALL)
    
    for match in stage_matches:
        stage_name = match.group(1).lower()
        similarity = float(match.group(2))
        content = match.group(3)
        
        # Extract areas for improvement
        improvements = []
        improvement_pattern = r"• ([^:]+): ([\d.]+)/100 - (.*?)(?=\n• |$)"
        for imp_match in re.finditer(improvement_pattern, content, re.DOTALL):
            improvements.append({
                "feature": imp_match.group(1),
                "score": float(imp_match.group(2)),
                "description": imp_match.group(3).strip()
            })
        
        sections["stages"][stage_name] = {
            "similarity": similarity,
            "improvements": improvements
        }
    
    # Extract detailed recommendations
    if "DETAILED RECOMMENDATIONS:" in text:
        rec_section = text.split("DETAILED RECOMMENDATIONS:")[1].split("PRACTICE DRILLS:")[0]
        for line in rec_section.split("\n"):
            if line.strip() and not line.startswith("---"):
                sections["detailed_recommendations"].append(line.strip())
    
    # Extract practice drills
    if "PRACTICE DRILLS:" in text:
        drills_section = text.split("PRACTICE DRILLS:")[1].split("PROGRESS TRACKING:")[0]
        current_drill = None
        for line in drills_section.split("\n"):
            if line.strip() and not line.startswith("---"):
                if line[0].isdigit():
                    if current_drill:
                        sections["practice_drills"].append(current_drill)
                    current_drill = {"name": line.strip(), "steps": []}
                elif current_drill and line.strip().startswith("•"):
                    current_drill["steps"].append(line.strip()[2:])
        if current_drill:
            sections["practice_drills"].append(current_drill)
    
    # Extract progress tracking
    if "PROGRESS TRACKING:" in text:
        tracking_section = text.split("PROGRESS TRACKING:")[1]
        for line in tracking_section.split("\n"):
            if line.strip() and not line.startswith("---"):
                sections["progress_tracking"].append(line.strip()[2:])
    
    return sections

@app.get("/")
async def root():
    return {"message": "Welcome to Shot Analysis API"}

@app.post("/analyze-shot")
async def analyze_shot(file: UploadFile = File(...), user_name: str = "user"):
    """
    Analyze a basketball shot from an uploaded video using the analyze_shot.py script
    """
    user_shot_path = USERS_SHOTS_DIR / file.filename
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a video file (mp4, mov, or avi)"
            )

        logger.info(f"Processing file: {file.filename} for user: {user_name}")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Users shots directory: {USERS_SHOTS_DIR}")
        logger.info(f"Results directory: {RESULTS_DIR}")
        logger.info(f"Models directory: {MODELS_DIR}")
        logger.info(f"Reference directory: {REFERENCE_DIR}")
        logger.info(f"Processed directory: {PROCESSED_DIR}")
        
        # Save the uploaded file
        contents = await file.read()
        with open(user_shot_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved file to: {user_shot_path}")
        
        # Run the analyze_shot.py script
        script_path = BASE_DIR / "bin" / "analyze_shot.py"
        cmd = [
            "python",
            str(script_path),
            str(user_shot_path),
            "--user", user_name,
            "--method", "ml"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            env=os.environ.copy()  # Pass environment variables to subprocess
        )
        
        logger.info(f"Script stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Script stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis script failed: {result.stderr}"
            )
        
        # Find the most recent result files
        result_json_files = list(RESULTS_DIR.glob(f"{user_name}_*_analysis.json"))
        result_txt_files = list(RESULTS_DIR.glob(f"{user_name}_*_recommendations.txt"))
        
        logger.info(f"Looking for result files in: {RESULTS_DIR}")
        logger.info(f"Found JSON files: {result_json_files}")
        logger.info(f"Found TXT files: {result_txt_files}")
        
        if not result_json_files or not result_txt_files:
            # Check if the results directory exists and is accessible
            if not RESULTS_DIR.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Results directory does not exist: {RESULTS_DIR}"
                )
            
            # List all files in the results directory
            all_files = list(RESULTS_DIR.glob("*"))
            logger.info(f"All files in results directory: {all_files}")
            
            raise HTTPException(
                status_code=500,
                detail=f"No analysis results found. Check if analyze_shot.py is saving results to: {RESULTS_DIR}"
            )
        
        latest_json = max(result_json_files, key=lambda x: x.stat().st_mtime)
        latest_txt = max(result_txt_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Using JSON file: {latest_json}")
        logger.info(f"Using TXT file: {latest_txt}")
        
        # Load the analysis results and recommendations
        with open(latest_json, 'r') as f:
            analysis_data = json.load(f)
            
        with open(latest_txt, 'r') as f:
            recommendations_text = f.read()
        
        # Parse recommendations into structured format
        structured_recommendations = parse_recommendations(recommendations_text)
        
        # Create a simplified analysis response
        simplified_analysis = {
            "overall_similarity": analysis_data["overall_similarity"],
            "stages": {}
        }
        
        for stage_name, stage_data in analysis_data["stages"].items():
            simplified_analysis["stages"][stage_name] = {
                "similarity_score": stage_data["similarity_score"],
                "key_metrics": {
                    feature: {
                        "value": data["value"],
                        "target": data["reference_mean"],
                        "score": data["score"]
                    }
                    for feature, data in stage_data["feature_analysis"].items()
                }
            }
        
        return {
            "filename": file.filename,
            "user_name": user_name,
            "analysis": simplified_analysis,
            "recommendations": structured_recommendations,
            "analysis_method": "ml"
        }
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis: {str(e)}"
        )
    finally:
        # Clean up the uploaded file
        if user_shot_path.exists():
            user_shot_path.unlink()
            logger.info(f"Cleaned up file: {user_shot_path}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 