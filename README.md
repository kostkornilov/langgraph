# Описание

Репозиторий содержит RAG-систему на базе LangChain, FAISS и Google GenAI эмбеддингов. Приложение генерирует ответ,основываясь на предоставленных PDF-файлах и предоставляет ссылки на файл в ответе. 

## Структура репозитория
- `books/`: исходные PDF.
- `data/`: сгенерированные артефакты (`chunks.jsonl`, `faiss/` индекс, метаданные, `embedding_config.json`).
- `scripts/`: утилиты (`ingest_books.py`, `build_faiss.py`, `check_retrieval.py`).
- `web/`: статический UI (`index.html`).
- `rag/`: общие помощники (например, `embedding_provider.py`).
- `server.py`: FastAPI-приложение, которое загружает FAISS, отвечает на запросы и вызывает LLM.
- `requirements.txt`: список зависимостей. 
- `docker-compose.yml` / `Dockerfile`: настройки контейнеров для запуска сервера.

## Перед запуском
- Windows PowerShell. Создать своё окружение (`python -m venv .venv`).
- `.env` должен содержать минимум:
  - `GOOGLE_API_KEY` (обязательно для `build_faiss.py`).
  - Опционально: `GOOGLE_EMBEDDING_MODEL`, `MODEL_ID` / `GOOGLE_MODEL_ID`, `ALLOW_EMBEDDING_FALLBACK`, `OPENAI_API_KEY`, `SENTENCE_TRANSFORMERS_MODEL`.
- Установленные зависимости: `pip install -r requirements.txt`.

## Запуск через PowerShell
1. **Активируйте виртуальное окружение**
   ```powershell
   .\myenv\Scripts\Activate.ps1
   ```
2. **Установите зависимости**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Инжест PDF в чанки**
   ```powershell
   python .\scripts\ingest_books.py
   ```
   - Получите `data/chunks.jsonl` с перекрывающимися чанками (~4800 символов, перекрытие 600).
4. **Постройте FAISS индекс**
   ```powershell
   $Env:GOOGLE_API_KEY = "<ваш ключ>"
   $Env:GOOGLE_EMBEDDING_MODEL = "text-embedding-004"  # при необходимости
   python .\scripts\build_faiss.py
   ```
   - Скрипт создаёт `data/faiss/index.faiss`, `metadata.jsonl` и `embedding_config.json`.
   - Требуются Google эмбеддинги, если не выставлен `ALLOW_EMBEDDING_FALLBACK`.
5. **Запустите FastAPI-сервер**
   ```powershell
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Откройте `http://localhost:8000` для `web/index.html`.
   - PDF остаются доступными по `/pdf/<имя>.pdf` с `#page=`.
6. **Проверки здоровья**
   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   Invoke-RestMethod http://localhost:8000/ready
   ```
7. **Примеры API**
   ```powershell
   Invoke-RestMethod http://localhost:8000/api/search -Method POST -ContentType 'application/json' -Body (ConvertTo-Json @{query='Раскольников'; top_k=3})
   Invoke-RestMethod http://localhost:8000/api/chat -Method POST -ContentType 'application/json' -Body (ConvertTo-Json @{thread_id='demo'; query='Кто такой Раскольников?'; top_k=3})
   ```

## Запуск через Docker
1. **Сборка и запуск**
   ```powershell
   docker-compose build
   docker-compose up
   ```
   - Убедитесь, что `books/` и `data/` примонтированы (`docker-compose.yml`).
   - Контейнер устанавливает зависимости, пересчитывает индекс при необходимости и слушает порт 8000.
2. **Принудительное пересоздание индекса внутри контейнера**
   ```powershell
   docker compose exec <service_name> python scripts/ingest_books.py
   docker compose exec <service_name> python scripts/build_faiss.py
   ```
   - Замените `<service_name>` на службу из `docker-compose.yml` (например, `app`).

## После изменений в корпусе
- Если обновили `books/`, заново запустите `ingest_books.py`, затем `build_faiss.py` и перезапустите сервер.
- `server.py` читает `data/faiss/embedding_config.json`, чтобы использовать тот же эмбеддер, что и индекс.

## Советы по отладке
- `Index not found` → удалите `data/faiss/` и повторите инжест + build.
- FAISS отдаёт метаданные вместо текста → проверьте, что и индекс, и запросы используют один эмбеддер (`scripts/check_retrieval.py`).
- `uvicorn` не стартует → смотрите лог, возможно, отсутствуют зависимости или env vars.

