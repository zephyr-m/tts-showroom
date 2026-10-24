FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY server.py index.html ./

ENV TTS_LIBS_READY=1 \
    HOST=0.0.0.0 \
    PORT=8765

EXPOSE 8765

CMD ["python", "server.py"]
