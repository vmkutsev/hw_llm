FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
WORKDIR /work
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt matplotlib==3.9.4
COPY . .
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
