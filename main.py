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

def extract_all_numbers(text: str) -> str:
    patterns = [
        r'((?:\d+[а-яa-z0-9\-]*(?:,\s*)?)+)\s*маршрут',
        r'номер\s*автобуса\s*(\d+[а-яa-z0-9\-]*)',
        r'(?:в|на)\s+автобусе\s*(?:№\s*)?(\d+[а-яa-z0-9\-]*)',
        r'(\d+[а-яa-z0-9\-]*)\s*автобус',
        r'(?:м/а|а/м|ма)\s*(\d+[а-яa-z0-9\-]*)',
        r'(\d+[а-яa-z0-9\-]*)\s*(?=м/а|а/м|ма)',
        r'маршрут(?:\s*автобуса)?\s*(?:№\s*)?(\d+[а-яa-z0-9\-]*)',
        r'№\s*(\d+[а-яa-z0-9\-]*)',
        r'(\d+[а-яa-z0-9\-]*)-го\s*маршрут',
        r'(?:кюар|KYUAR)\s*(\d+)',
    ]
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        if found:
            for item in found:
                if ',' in item:
                    parts = re.split(r',\s*', item)
                    matches.extend(parts)
                else:
                    matches.append(item)
    seen = set()
    unique_matches = []
    for m in matches:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            unique_matches.append(m)
    return ", ".join(unique_matches) if unique_matches else None

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
        numbers = extract_all_numbers(complaint)
        results.append({
            "complaint": complaint,
            "cluster": cluster,
            "category": categories.get(cluster, "Неизвестная категория"),
            "route": numbers if numbers else "Не найден"
        })
    
    return templates.TemplateResponse("results.html", {"request": request, "results": results})
