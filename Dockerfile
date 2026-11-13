# Parley editor. The weights are not in the image: mount them at /weights.
#   docker build -t parley-editor .
#   docker run --rm -v /path/to/weights:/weights -v "$PWD":/work parley-editor parley edit --device cpu --image /work/x.png --mask /work/m.png --text "pull the top in a little" --out /work/new.png
# Demo, reachable only from this machine:
#   docker run --rm -p 127.0.0.1:7860:7860 -e PARLEY_DEMO_HOST=0.0.0.0 -v /path/to/weights:/weights parley-editor parley demo --device cpu
# GPU: add --gpus all and use --device cuda.
# CPU-only image, much smaller, built with the CPU wheels of the same torch version:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t parley-editor:cpu .
FROM python:3.11-slim

ARG TORCH_INDEX=
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/parley
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN if [ -n "$TORCH_INDEX" ]; then pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" torch==2.14.1+cpu torchaudio==2.11.0+cpu torchvision==0.29.1+cpu ; fi \
    && pip install --no-cache-dir ".[demo]"

ENV PARLEY_WEIGHTS=/weights
RUN useradd --create-home parley
USER parley
WORKDIR /home/parley
CMD ["parley", "--help"]
