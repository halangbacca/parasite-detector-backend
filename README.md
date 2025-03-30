# 🧠 Backend FastAPI - Detecção de Ovos com YOLOv11

Este é o backend do sistema de detecção de ovos de parasitas usando **FastAPI** e **YOLOv11** (Ultralytics). Ele permite processar **imagens ou vídeos**, detectar ovos com base em diferentes modelos YOLO, gerar **relatórios em PDF/CSV**, e retornar o arquivo anotado para visualização no frontend.

## 🚀 Funcionalidades

- ✅ Processamento de **imagens** e **vídeos**
- 🤖 Suporte a múltiplos modelos YOLOv11 (`yolov11n`, `s`, `m`, `l`, `x`)
- 🎯 Ajuste do limiar de confiança via parâmetro `threshold`
- 🧮 Contagem de ovos por classe (evita contagem duplicada em vídeo com IoU)
- 🖼️ Retorno de imagem/vídeo com caixas desenhadas
- 📄 Geração de relatórios em **CSV** e **PDF**
- 🔗 Endpoints para download dos arquivos gerados

## 📦 Requisitos

- Python 3.8+
- Modelos YOLOv11 (.pt) salvos na pasta `weights/`
- Dependências instaladas (veja abaixo)

## 📁 Estrutura de Pastas

```
weights/           # Modelos YOLOv11 (.pt)
output_images/     # Imagens anotadas
output_videos/     # Vídeos anotados
output_reports/    # Relatórios CSV e PDF
```

## 📥 Instalação

```bash
git clone https://github.com/halangbacca/parasite-detector-backend.git
cd parasite-detector-backend

# Crie um ambiente virtual
python -m venv detector
source detector/bin/activate  # Windows: detector\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Exemplo de `requirements.txt`

```txt
fastapi==0.115.12
uvicorn==0.34.0
pandas==2.2.3
fpdf==1.7.2
opencv-python==4.11.0.86
ultralytics==8.3.98
python-multipart==0.0.20
```

## ▶️ Execução

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: [http://localhost:8000](http://localhost:8000)

## 🔁 Endpoints Principais

### `POST /predict/`

Processa uma imagem ou vídeo enviado via `multipart/form-data`.

**Parâmetros Query:**

- `threshold`: confiança mínima (ex: `0.5`)
- `model`: modelo YOLOv11 desejado (ex: `yolov11n`)

**Resposta:**

```json
{
  "detections_count": {"classe1": 3, "classe2": 1},
  "download_url": "http://localhost:8000/download/nome_do_arquivo.mp4",
  "csv_url": "http://localhost:8000/download/relatorio.csv",
  "pdf_url": "http://localhost:8000/download/relatorio.pdf"
}
```

### `GET /download/{filename}`

Retorna um arquivo gerado anteriormente: imagem, vídeo, PDF ou CSV.

---

## 🔒 CORS

Liberado para: `http://localhost:8081` (frontend em Vue)

## 📌 Observações

- A detecção em vídeo evita contagens duplicadas usando IoU.
- Os arquivos gerados são salvos com timestamp para evitar conflitos.
- Todos os diretórios são criados automaticamente, se não existirem.

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE).
