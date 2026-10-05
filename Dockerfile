FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tracks are resampled for speech-to-text with soxr: fail the build, not a session, without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libopus0 \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -buildconf | grep -q -- --enable-libsoxr

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["scrollkeeper"]
