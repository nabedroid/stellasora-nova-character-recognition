FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# ultralytics が使う opencv のために必要なシステムライブラリ (runtime イメージには入っていない)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pillow ultralytics
WORKDIR /workspace
CMD ["bash"]
