# Запуск локально (без Docker)

Этот файл описывает минимальные шаги для запуска проекта локально на Windows (PowerShell) без использования Docker. Предполагается, что в репозитории есть виртуальное окружение `myenv` (включено в репо) или вы можете создать своё.

Основные шаги:
- активировать виртуальное окружение
- установить зависимости
- выполнить инжест PDF → чанки
- построить FAISS-индекс
- запустить FastAPI сервер

Важные директории и артефакты:
- `books/` — исходные PDF
- `data/chunks.jsonl` — сгенерированные чанки (тексты)
- `data/faiss/index.faiss` и `data/faiss/metadata.jsonl` — FAISS индекс + метаданные
- `data/server_conversations.sqlite` — SQLite база разговоров (если используется)

Переменные окружения (ключевые):
- `GOOGLE_API_KEY` — **обязательно** для построения индекса (используем Google text-embedding-004)
- `GOOGLE_EMBEDDING_MODEL` — опционально, модель эмбеддингов (по умолчанию `text-embedding-004`)
- `MODEL_ID` / `GOOGLE_MODEL_ID` — id модели для генерации (если сервер должен вызывать LLM)
- `OPENAI_API_KEY` — fallback, только если включите режим ALLOW_EMBEDDING_FALLBACK
- `SENTENCE_TRANSFORMERS_MODEL` — локальная модель (используется лишь в fallback-сценариях)

1) Активировать виртуальное окружение

Если вы будете использовать встроенное `myenv` в репозитории:

```powershell
.\myenv\Scripts\Activate.ps1
```

Если хотите создать новое окружение и активировать его:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Установить зависимости

(если вы используете `myenv`, пакеты уже могут быть установлены — команда безопасна)

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Инжест PDF → чанки

Этот шаг извлекает текст из PDF в `books/` и формирует `data/chunks.jsonl`. Выполняйте при изменении/добавлении PDF или если chunks не созданы.

```powershell
python .\scripts\ingest_books.py
# Файл data/chunks.jsonl будет создан/перезаписан
```

Примечание: теперь используется детерминированный символьный сплиттер (~4800 символов с перекрытием 600), что улучшает RAG по русским PDF.

4) Построить FAISS-индекс

Этот шаг вычисляет эмбеддинги для чанков и сохраняет индекс в `data/faiss/`. Используется Google GenAI, поэтому обязательно установите ключ перед запуском.

```powershell
$Env:GOOGLE_API_KEY = "<ваш ключ>"
$Env:GOOGLE_EMBEDDING_MODEL = "text-embedding-004"   # необязательно
python .\scripts\build_faiss.py
# Результат: data/faiss/index.faiss, metadata.jsonl и embedding_config.json
```

Параметры и поведение:
- Скрипт требует Google API; fallback на локальные модели отключён по умолчанию, чтобы не смешивать пространства эмбеддингов.
- Если вам всё-таки нужен fallback (например, без доступа к Google), установите `ALLOW_EMBEDDING_FALLBACK=1`, но после переключения обязательно пересоберите индекс и убедитесь, что сервер использует ту же модель.
- Эмбеддинги вычисляются один раз при запуске скрипта. Сервер при старте читает `embedding_config.json`, чтобы поднять тот же провайдер.

5) Запуск сервера (FastAPI)

Запустите Uvicorn чтобы поднять API:

```powershell
# usando virtualenv активирован
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

После запуска:
- Откройте в браузере http://localhost:8000 — загрузится статическая фронтенд-страница (если она есть).
- PDF доступны по URL: `http://localhost:8000/pdf/<имя_файла>.pdf` (фронтенд использует ссылки с `#page=` для перехода к страницам).

6) Проверка здоровья и готовности

Если в проект добавлены эндпойнты `/health` и `/ready`, можно проверить их:

```powershell
# Liveness
Invoke-RestMethod -Uri http://localhost:8000/health -Method GET
# Readiness (проверяет, загрузился ли FAISS индекс)
Invoke-RestMethod -Uri http://localhost:8000/ready -Method GET
```

7) Быстрое тестирование поиска и чата (пример)

# Поиск (returns top results metadata)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/search -Method POST -ContentType 'application/json' -Body (ConvertTo-Json @{query='Raskolnikov'; k=3})
```

# Чат — если настроена модель (MODEL_ID), сервер отправит retrieved chunks и вопрос модели
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method POST -ContentType 'application/json' -Body (ConvertTo-Json @{thread_id='demo'; query='Кто такой Раскольников?'; top_k=3})
```

8) Что делать, если нужно принудительно пересчитать эмбеддинги

По умолчанию пересчёт — это отдельная операция (`scripts/build_faiss.py`). Чтобы не терять время и ресурсы, не запускайте её при каждом старте. Если вы часто обновляете `books/`, добавьте в CI или локальный шаг, который вызывает `build_faiss.py` по необходимости.

Если хотите, я могу добавить в `server.py` безопасный endpoint `/admin/rebuild-index` (защищённый паролем или переменной окружения) который будет запускать пересчёт на запрос.

9) Советы по производительности и отладке

- Если у вас мало RAM, выбирайте меньшую `batch_size` при вычислении эмбеддингов в `scripts/build_faiss.py` или используйте меньшую модель (например `all-MiniLM-L6-v2` уже по умолчанию).
- Если вы используете GPU, убедитесь, что `torch` и необходимые CUDA-драйверы установлены и `sentence-transformers` использует CUDA.
- Если видите ошибки при загрузке FAISS, проверьте, что `data/faiss/index.faiss` и `data/faiss/metadata.jsonl` существуют и не повреждены.

10) Частые проблемы и решения

- "Index not found" — убедитесь, что `data/faiss` существует и пути корректные. Запустите `build_faiss.py` вручную.
- "Out of memory" при построении эмбеддингов — уменьшите batch size или используйте менее тяжёлую модель.
- Ошибки сетевых ключей API — проверьте, что `GOOGLE_API_KEY` / `OPENAI_API_KEY` установлены в окружении перед запуском процесса, если вы используете облачные эмбеддинги.

11) Дополнительно (опции)

- Локальный dev: используйте `--reload` в uvicorn для авто-перезагрузки.
- Для продакшена: рекомендуем поместить сервер за обратным прокси (nginx), настроить TLS и организовать бэкап `data/`.

Если хотите, могу добавить:
- `.env.example` с примерами переменных окружения (я могу добавить сейчас),
- безопасный endpoint для пересчёта индекса,
- или скрипт/инструкцию для автоматического построения индекса в CI.

---

Если нужно — переведу эти инструкции в `README.md` или интегрирую в `DEPLOY.md`.