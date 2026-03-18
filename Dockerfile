FROM docker:cli AS dockercli
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libopus0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

CMD ["scrollkeeper"]
