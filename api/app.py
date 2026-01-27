from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from api.model import load_model, predict

app = FastAPI(title="Sugarcane Leaf Disease Detection API")

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = predict(model, img_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)