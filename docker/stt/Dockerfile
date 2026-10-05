# Models are fetched (and checksum-verified) in a separate stage so the archive never lands in
# the final image. They are baked in so the pod needs no network or volume to start.
FROM python:3.12-slim AS models
COPY fetch_models.py /tmp/fetch_models.py
RUN python /tmp/fetch_models.py /models

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /models /models
COPY pipeline.py app.py ./

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
