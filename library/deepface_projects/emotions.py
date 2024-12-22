def analyze(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_filename = tmp.name
            tmp.write(file.read())

        obj = DeepFace.analyze(
            img_path = temp_filename,
            actions = ['age', 'gender', 'race', 'emotion']
        )



        return {"result": obj}

    except Exception as e:
        logging.error("An error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

