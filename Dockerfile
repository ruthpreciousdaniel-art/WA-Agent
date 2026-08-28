FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Create data dirs inside the image instead of COPYing an (often empty) folder.
# Your app also creates these at startup via config.py's mkdir(parents=True, exist_ok=True),
# so this line is really just belt-and-suspenders.
RUN mkdir -p /code/data/uploads /code/data/faiss_index

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
