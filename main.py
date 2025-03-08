from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar o modelo YOLOv11
model = YOLO("/home/halan/Área de trabalho/jupyter_notebook/runs/detect/train10/weights/best.pt")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler a imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Fazer a inferência
        results = model(img)

        # Desenhar as detecções na imagem
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = result.names[int(box.cls)]
                conf = float(box.conf)

                # **Aumentando ainda mais a espessura e o tamanho do texto**
                box_thickness = 6  # Espessura da bounding box (antes 3)
                font_scale = 0.8  # Tamanho da fonte (antes 1.2)
                font_thickness = 2  # Espessura da fonte (antes 2)

                # Desenhar bounding box bem visível
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                # Criar fundo preto para o texto (melhor legibilidade)
                (w, h), _ = cv2.getTextSize(f"{class_name} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            font_thickness)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 255, 0), -1)  # Fundo verde

                # Adicionar texto sobre a bounding box
                cv2.putText(img, f"{class_name} {conf:.2f}", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Codificar a imagem para enviar ao frontend
        _, encoded_img = cv2.imencode(".jpg", img)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}
