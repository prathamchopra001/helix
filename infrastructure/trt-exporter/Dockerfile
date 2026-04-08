# TensorRT export container
# Base: NVIDIA TensorRT 8.6.1 + CUDA 12.2 + Python 3.10
# Only used for model export — never serves traffic
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Install PyTorch 2.1.0 with CUDA 12.1 (compatible with CUDA 12.2 runtime)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install ONNX and runtime
RUN pip install --no-cache-dir \
    onnx==1.14.0 \
    onnxruntime-gpu==1.16.0

# Install project packages
COPY shared/ /opt/helix/shared/
RUN pip install --no-cache-dir /opt/helix/shared/

COPY services/training/ /opt/helix/training/
RUN pip install --no-cache-dir \
    "mlflow>=2.11.0,<2.18.0" \
    minio>=7.2.0 \
    psycopg2-binary>=2.9.9 \
    scikit-learn>=1.4.0 \
    pandas>=2.0.0

# Export entrypoint
COPY scripts/export_tensorrt.sh /export.sh
RUN chmod +x /export.sh

WORKDIR /opt/helix
ENV PYTHONPATH="/opt/helix/shared/src:/opt/helix/training/src"

ENTRYPOINT ["/export.sh"]
