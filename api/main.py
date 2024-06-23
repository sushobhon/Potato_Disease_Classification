from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

MODEL = tf.keras.models.load_model('../Saved_Models/potatoes.h5')
# endpoint = 
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Defining a function read a image and return a numpy array
def read_file_as_image(data) -> np.ndarray:
    # Fist we convert data to bytes, then convert to In=mage class then convert to numpy array
    image = np.array(Image.open(BytesIO(data))) 
    
    # Resize the image to (256, 256, 3) shape
    image = np.array(Image.fromarray(image).resize((256, 256)))

    return image
    

@app.get('/ping')
async def ping():
    return "Hello I am Live!!"

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    # Adding one extra dimention as our image acepts a batch of image and we are providing single image
    img_batch = np.expand_dims(image, 0)


    # Predicting
    prediction = MODEL.predict(img_batch)

    # Prediction is for a batch of image. Hence taking preiction[0] as the prediction of simgle image
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)
    