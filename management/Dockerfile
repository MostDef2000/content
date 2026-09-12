FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY management/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY management/main.py /workspace/management/main.py
COPY management/static /workspace/management/static

EXPOSE 8000

CMD ["uvicorn", "management.main:app", "--host", "0.0.0.0", "--port", "8000"]
