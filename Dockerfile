FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY README.md pyproject.toml ./
COPY src ./src

# CPU-only torch wheel keeps the image small; the smoke test does not need
# CUDA. Override with a CUDA wheel for GPU training or inference.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.6.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
       numpy==1.26.4 scipy==1.13.0 h5py==3.11.0 tqdm==4.66.4 \
    && pip install --no-cache-dir --no-deps .

# Default to the offline smoke test (no data, no checkpoint required).
CMD ["monkey", "smoke"]
