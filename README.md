# FastAPI Complaint Classification

## Описание проекта
Этот проект представляет собой **API-сервис** на **FastAPI**, который классифицирует жалобы на основе модели **DeepPavlov/rubert-base-cased**.

🔹 **Что делает программа?**
1. Загружает **XLSX-файл** с жалобами.
2. Читает жалобы из **колонки "Описание"**.
3. Использует **нейросеть** для классификации текста.
4. Выводит результат в виде таблицы с **3 колонками**:
   - **Жалоба** (текст)
   - **Кластер** (номер категории)
   - **Категория** (название категории)

---

## Установка зависимостей
Перед запуском **установите все библиотеки**, необходимые для проекта:

```bash
pip install -r requirements.txt

---

## Запуск сервера FastAPI 
Для старта сервера выполните:

```bash
uvicorn main:app --reload
