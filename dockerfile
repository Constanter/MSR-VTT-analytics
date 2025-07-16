FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && \
    apt-get install -y ffmpeg

COPY . .

CMD ["python", "run_pipeline.py"]