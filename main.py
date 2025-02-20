import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_path = "C:\\Users\danny\\My_Folder\\aitu_project\\RP1\\MODEL_TESTS\\PROJECTS\\complaints_ml_models\\DeepPavlov\\rubert_model"
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

def classify_complaint(complaint: str) -> int:
    inputs = tokenizer(
        complaint,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

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

    results = []
    for idx, row in data.iterrows():
        complaint = row["Описание"]
        if pd.isna(complaint):
            continue
        cluster = classify_complaint(str(complaint))
        results.append({
            "complaint": complaint,
            "cluster": cluster,
            "category": categories.get(cluster, "Неизвестная категория")
        })

    return templates.TemplateResponse("results.html", {"request": request, "results": results})
