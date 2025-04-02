from fastapi import FastAPI, UploadFile, File, Query, WebSocket
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from collections import defaultdict
from fpdf import FPDF
import pandas as pd
import cv2
import tempfile
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

IMG_SIZE = (1024, 768)
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_IMAGE_DIR = "output_images"
OUTPUT_REPORT_DIR = "output_reports"
WEIGHTS_DIR = "weights"

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

executor = ThreadPoolExecutor()
fila_progresso = asyncio.Queue()


@app.websocket("/ws/progresso")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            progresso = await fila_progresso.get()
            await websocket.send_text(str(progresso))
    except:
        await websocket.close()


def salvar_relatorio_csv(detections: dict, filename: str):
    df = pd.DataFrame(list(detections.items()), columns=["Classe", "Quantidade"])
    csv_path = f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def salvar_relatorio_pdf(detections: dict, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Relatório de Detecção", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    for classe, qtd in detections.items():
        pdf.cell(200, 10, txt=f"{classe}: {qtd} ovos detectados", ln=True)
    pdf_path = f"{filename}.pdf"
    pdf.output(pdf_path)
    return pdf_path


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0


def processar_video_em_thread(video_path, out_path, yolo_model, threshold, detections_count, total_frames, loop):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    seen_boxes = {class_name: [] for class_name in yolo_model.names.values()}
    iou_threshold = 0.5
    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf < threshold:
                    continue
                class_name = result.names[int(box.cls.item())]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                current_box = (x1, y1, x2, y2)
                if all(iou(current_box, b) < iou_threshold for b in seen_boxes[class_name]):
                    seen_boxes[class_name].append(current_box)
                    detections_count[class_name] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        out.write(frame)
        processed += 1
        progresso = int((processed / total_frames) * 100)
        asyncio.run_coroutine_threadsafe(fila_progresso.put(progresso), loop)

    cap.release()
    out.release()


@app.post("/predict/")
async def predict(
        file: UploadFile = File(...),
        threshold: float = Query(0.5),
        model: str = Query("YOLO11n")
):
    try:
        model_path = os.path.join(WEIGHTS_DIR, f"{model}.pt")
        if not os.path.exists(model_path):
            return JSONResponse(content={"error": f"Modelo '{model}' não encontrado."}, status_code=400)

        yolo_model = YOLO(model_path)

        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        detections_count = defaultdict(int)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = os.path.splitext(os.path.basename(file.filename))[0]

        if suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames == 0:
                os.remove(tmp_path)
                return JSONResponse(content={"error": "Vídeo inválido."}, status_code=400)

            output_filename = f"{filename_base}_{timestamp}_out.mp4"
            out_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                processar_video_em_thread,
                tmp_path,
                out_path,
                yolo_model,
                threshold,
                detections_count,
                total_frames,
                loop
            )

            csv_path = salvar_relatorio_csv(detections_count,
                                            os.path.join(OUTPUT_REPORT_DIR, filename_base + "_" + timestamp))
            pdf_path = salvar_relatorio_pdf(detections_count,
                                            os.path.join(OUTPUT_REPORT_DIR, filename_base + "_" + timestamp))

            os.remove(tmp_path)
            return JSONResponse(content={
                "detections_count": detections_count,
                "download_url": f"http://localhost:8000/download/{output_filename}",
                "csv_url": f"http://localhost:8000/download/{os.path.basename(csv_path)}",
                "pdf_url": f"http://localhost:8000/download/{os.path.basename(pdf_path)}"
            })

        elif suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            img = cv2.imread(tmp_path)
            img_resized = cv2.resize(img, IMG_SIZE)
            results = yolo_model(img_resized)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf.item())
                    if conf < threshold:
                        continue
                    class_name = result.names[int(box.cls.item())]
                    detections_count[class_name] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_resized, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"{filename_base}_{timestamp}_out.jpg")
            cv2.imwrite(output_image_path, img_resized)

            await fila_progresso.put(100)

            csv_path = salvar_relatorio_csv(detections_count,
                                            os.path.join(OUTPUT_REPORT_DIR, filename_base + "_" + timestamp))
            pdf_path = salvar_relatorio_pdf(detections_count,
                                            os.path.join(OUTPUT_REPORT_DIR, filename_base + "_" + timestamp))

            os.remove(tmp_path)
            return JSONResponse(content={
                "detections_count": detections_count,
                "download_url": f"http://localhost:8000/download/{os.path.basename(output_image_path)}",
                "csv_url": f"http://localhost:8000/download/{os.path.basename(csv_path)}",
                "pdf_url": f"http://localhost:8000/download/{os.path.basename(pdf_path)}"
            })

        else:
            os.remove(tmp_path)
            return JSONResponse(content={"error": "Tipo de arquivo não suportado"}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    for folder in [OUTPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR, OUTPUT_REPORT_DIR]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".mp4":
                return StreamingResponse(
                    open(path, "rb"),
                    media_type="video/mp4",
                    headers={"Content-Disposition": f"inline; filename={filename}"}
                )
            elif ext == ".jpg":
                return FileResponse(path, media_type="image/jpeg", filename=filename)
            elif ext == ".csv":
                return FileResponse(path, media_type="text/csv", filename=filename)
            elif ext == ".pdf":
                return FileResponse(path, media_type="application/pdf", filename=filename)
            else:
                return FileResponse(path, media_type="application/octet-stream", filename=filename)

    return JSONResponse(content={"error": "Arquivo não encontrado."}, status_code=404)
