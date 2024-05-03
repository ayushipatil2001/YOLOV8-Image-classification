
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

app = FastAPI()

def personality(image):
    model = YOLO('.//runs//classify//train//weights//best.pt')

    img = Image.open(BytesIO(image))
    img = img.resize((244, 244))

    
    results = model(img, show=True)

   
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    prediction = names_dict[probs.index(max(probs))]

    return prediction

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Image</title>
    </head>
    <body>
        <h1>Upload image to predict personality trait</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    prediction = personality(contents)
    return {"prediction": prediction}
