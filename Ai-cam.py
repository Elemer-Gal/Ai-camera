import cv2
from ultralytics import YOLO
import time
import requests
import base64 
#change to model path, Full path works best and reduces room for error
model = YOLO(r"")
#camera on
cap = cv2.VideoCapture(0)
#yolo timing and frames
last_frame = []
last = 0
cooldown = 1.0
#moondream, response, and timings
last_description = ""
last_moondream = 0
moondream_delay = 9.0
#calling moondream
def run_moondream(frame, prompt):
    _, buf = cv2.imencode(".jpg",frame)
    image_b64 = base64.b64encode(buf).decode()
    #model payload
    payload = {
        
        "model": "moondream:1.8b",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    r = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout = 120
    )
    return r.json()[ "response"]

while True:
    success,frame = cap.read()
    #check if cam good
    if not success:
        break
    now = time.time()
    #for timing if last is more than cooldown it does whats below
    if now - last > cooldown:
        results = model(frame, verbose = False, save = False)
        last = now
        last_frame = []
        for r in results:
            for box in r.boxes:
                #setting up bounding boxes
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                last_frame.append((x1,y1,x2,y2,label,conf))
    #for moondream
    if last_frame and now - last_moondream > moondream_delay:
        #feeding it frames
        snapshot = frame.copy()
        last_moondream = now
        description = run_moondream(snapshot,prompt = "Describe what you see")
        last_description = description
        #print response from model
        print(last_description)

    #printing of boxes and labels and conf
    for (x1,y1,x2,y2,label,conf) in last_frame:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        text = f"{label} {conf:.2f}"

        cv2.putText(
        frame,
        text,
        (x1,y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2

    )
    cv2.imshow("Yolo", frame)
    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()