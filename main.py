from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from collections import defaultdict

app = FastAPI()

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Carregar o modelo YOLOv11
model = YOLO("./best.pt")

# Definir tamanho padrão da imagem para o YOLO
IMG_SIZE = (1024, 768)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: float = Query(0.5, description="Confiança mínima para detecção")):
    try:
        # Ler a imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Redimensionar a imagem para 640x640
        img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Fazer a inferência no YOLO
        results = model(img_resized)

        # Dicionário para contar os ovos detectados
        detections_count = defaultdict(int)

        # Desenhar as detecções na imagem com base no threshold escolhido pelo usuário
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.item())  # Converter tensor para float

                if conf < threshold:  # Aplicar threshold de confiança
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = result.names[int(box.cls.item())]  # Garantir que pegamos a classe correta

                # Contar os ovos detectados por classe
                detections_count[class_name] += 1

                # Ajustar bounding box e texto
                box_thickness = 6
                font_scale = 1.5
                font_thickness = 3

                # Desenhar bounding box
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                # Criar fundo para o texto
                (w, h), _ = cv2.getTextSize(f"{class_name} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(img_resized, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 255, 0), -1)

                # Adicionar texto
                cv2.putText(img_resized, f"{class_name} {conf:.2f}", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Codificar a imagem redimensionada e processada para enviar ao frontend
        _, encoded_img = cv2.imencode(".jpg", img_resized)

        # Retornar JSON com a contagem de ovos e a imagem em hexadecimal
        return JSONResponse(
            content={
                "detections_count": detections_count,
                "image": encoded_img.tobytes().hex()
            }
        )

    except Exception as e:
        return {"error": str(e)}
