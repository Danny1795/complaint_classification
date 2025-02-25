import io
import time
import re
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_path = "C:\\Users\\danny\\My_Folder\\aitu_project\\RP1\\MODEL_TESTS\\PROJECTS\\complaints_ml_models\\DeepPavlov\\rubert_model"
tokenizer_path = "C:\\Users\\danny\\My_Folder\\aitu_project\\RP1\\MODEL_TESTS\\PROJECTS\\complaints_ml_models\\DeepPavlov\\rubert_tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() 

categories = {
    0: "Проблемы с транспортом",
    1: "Проблемы с персоналом",
    2: "Инфраструктура остановок",
    3: "Неисправности оборудования",
    4: "Похвала",
    5: "Организационно-технические проблемы",
    6: "Прочие жалобы"
}

def extract_route_number(text: str) -> str:
    """
    Извлекает номер маршрута из текста жалобы по шаблонам:
      - 'м/а' или 'а/м', за которыми следует число,
      - символ '№' и число,
      - число, за которым следует слово 'маршрут'
    """
    patterns = [
        r'(?:м/а|а/м)\s*(\d+)',
        r'№\s*(\d+)',
        r'(\d+)\s*маршрут'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def classify_complaints_batch(complaints, batch_size=32):
    predictions = []
    start_time = time.perf_counter()  
    for i in range(0, len(complaints), batch_size):
        batch = complaints[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1).tolist()
        predictions.extend(batch_preds)
    elapsed_time = time.perf_counter() - start_time
    print(f"Классификация {len(complaints)} жалоб заняла {elapsed_time:.4f} секунд")
    return predictions

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def process_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка при чтении файла: {e}"})
    
    if "Описание" not in data.columns:
        return templates.TemplateResponse("index.html", {"request": request, "error": "В файле отсутствует колонка 'Описание'."})

    complaints = [str(row["Описание"]) for idx, row in data.iterrows() if pd.notna(row["Описание"])]
    predictions = classify_complaints_batch(complaints, batch_size=32)

    results = []
    for complaint, cluster in zip(complaints, predictions):
        route = extract_route_number(complaint)
        results.append({
            "complaint": complaint,
            "cluster": cluster,
            "category": categories.get(cluster, "Неизвестная категория"),
            "route": route if route else "Не найден"
        })

    return templates.TemplateResponse("results.html", {"request": request, "results": results})
