from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from api.model import load_model, predict as model_predict
from PIL import UnidentifiedImageError

app = FastAPI(title="Sugarcane Leaf Disease Detection API")

@app.on_event("startup")
async def startup_event():
    """
    Load the ML model into app state on startup.
    """
    app.state.model = load_model()

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict sugarcane leaf disease from an uploaded image.

    Args:
        file (UploadFile): Image file uploaded by the user.

    Returns:
        JSON: Predicted class and class probabilities.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(
            {"error": "Invalid file type. Only JPEG and PNG are allowed."},
            status_code=400,
        )

    try:
        img_bytes = await file.read()
        result = model_predict(app.state.model, img_bytes)
        return JSONResponse(result)

    except UnidentifiedImageError:
        return JSONResponse({"error": "Could not process image."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)