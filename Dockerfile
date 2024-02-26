FROM continuumio/miniconda3

# getting python 3.10
RUN conda install -y python=3.10
RUN ln -s /opt/conda/bin/python3.10 /usr/local/bin/python

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    tmux \
    python3-dev \
    libgdal-dev \
    libeccodes-dev \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Make RUN commands use the new environment:
RUN echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Demonstrate the environment is activated:
RUN echo "Make sure cfgrib is installed:"
RUN python -c "import pandapower"
