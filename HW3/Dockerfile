FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
WORKDIR /work
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
ENV HF_DISABLE_TELEMETRY=true TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY . .
CMD ["python", "run_notebook.py"]
