FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.31.0

COPY server.py index.html ./

ENV TTS_LIBS_READY=1 \
    HOST=0.0.0.0 \
    PORT=8765 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

EXPOSE 8765

CMD ["python", "server.py"]
