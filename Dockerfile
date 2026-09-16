FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# CPU-only torch: avoids ~2 GB of nvidia/cuda/triton wheels (sentence-transformers pulls torch)
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY src/ src/
COPY app/ app/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]