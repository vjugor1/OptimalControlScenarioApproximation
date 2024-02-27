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
    libglpk-dev python3.8-dev libgmp3-dev \
    glpk-utils libglpk-dev glpk-doc \ 
    gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Make RUN commands use the new environment:
RUN echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    conda install -c conda-forge ipopt=3.11.1

# Demonstrate the environment is activated:
RUN echo "Make sure cfgrib is installed:"
RUN python -c "import pandapower"
