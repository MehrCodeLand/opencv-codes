@app.post("/verify-face/")
async def verify_face():
    """API endpoint to verify if two images contain the same face"""
    try:
        result = DeepFace.verify(img1_path="davidmask.JPG", img2_path="davidmask2.JPG",detector_backend="retinaface", model_name="ArcFace" ,normalization="ArcFace")

        return {"result": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


