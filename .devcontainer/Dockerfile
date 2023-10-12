
# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Sets the home directory
WORKDIR /workspace

# Specifies that all commands should be run as root (important for the next steps)
USER root

# Install necessary packages
RUN apt-get update && apt-get install -y     git     curl     nano     build-essential     python3-pip     python3-dev     python3-setuptools     && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install -U pip setuptools wheel numpy pandas jupyter transformers torch nltk

# Starts Jupyter when the container is run, maps traffic from port 8888 in the container to the host
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
