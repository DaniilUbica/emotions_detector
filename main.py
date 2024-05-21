from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import cv2
import numpy as np
import base64

from model import guess_fer_emotion

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode('utf-8')

templates.env.filters['b64encode'] = b64encode

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, buffer = cv2.imencode('.jpg', img)
    processed_image = buffer.tobytes()

    caption = guess_fer_emotion(img)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_data": processed_image,
        "caption": caption
    })

@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        i = 0
        emotion = "can't understand emotion"
        while True:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if i == 7: # update emotion every 7 frames
                emotion = guess_fer_emotion(image)
                i = 0
            
            cv2.putText(image, emotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, jpeg_image = cv2.imencode('.jpg', image)
                
            jpeg_base64 = base64.b64encode(jpeg_image).decode('utf-8')
            jpeg_base64_str = f"data:image/jpeg;base64,{jpeg_base64}"
            i += 1

            await websocket.send_text(jpeg_base64_str)
    except WebSocketDisconnect:
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
