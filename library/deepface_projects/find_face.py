# import deepface , Json and ...

def find_face(file:UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_filename = tmp.name
            tmp.write(await file.read())

            dfs = DeepFace.find(
                img_path=temp_filename,
                db_path='my_sample_db_images',
            )
        logging.info(f"{dfs}")
        if isinstance(dfs, np.ndarray):
            dfs = dfs.tolist()
        elif isinstance(dfs, (np.generic, np.number)):
            dfs = dfs.item()

        return JSONResponse(content={"results": dfs})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

