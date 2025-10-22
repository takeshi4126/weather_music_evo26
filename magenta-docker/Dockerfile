FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    pkg-config \
    libasound2-dev \
    wget \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# NumPy 1.23.5 + TF2.10.0 + TFP 0.18.0 + Magenta (Minimum configuration)
RUN pip install --upgrade pip && \
    pip install numpy==1.23.5 && \
    pip install tensorflow==2.10.0 && \
    pip install tensorflow-probability==0.18.0 && \
    pip install note-seq==0.0.3 && \
    pip install tf_slim==1.1.0 && \
    pip install magenta==2.1.3 \
        --no-deps \
        --no-binary llvmlite,python-rtmidi,pretty-midi,numba \
        --no-cache-dir

# Copy the pre-trained model.
RUN mkdir -p /app/models
COPY ../models/basic_rnn.mag /app/models/basic_rnn.mag

# Replace a magenta source code to avoid inclusion of tensor2tensor, 
# which cannot be installed due to missing tensor-addons for ARM architecture. 
COPY training.py /usr/local/lib/python3.8/site-packages/magenta/contrib/training.py

# Output folder of the MIDI file.
RUN mkdir -p /app/output

ENTRYPOINT ["/usr/bin/bash"]
