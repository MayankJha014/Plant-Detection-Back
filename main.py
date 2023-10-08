from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI()


MODEL = tf.keras.models.load_model("./models/1")

CLASS_NAMES = ['Alstonia Scholaris',
               'Arjun',
               'Bael',
               'Basil',
               'Chinar',
               'Gauva',
               'Jamun',
               'Jatropha',
               'Lemon',
               'Mango',
               'Pomegranate',
               'Pongamia Pinnata']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    new_height = 400
    new_width = 600
    resized_image = tf.image.resize(image, [new_height, new_width])

    img_batch = np.expand_dims(resized_image, 0)
    # img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'detection': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
