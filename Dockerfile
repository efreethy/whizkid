FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git

# Upgrade pip
RUN pip3 install --upgrade pip

# Install required Python libraries
RUN pip3 install torch torchvision torchaudio transformers accelerate

# Set the working directory
WORKDIR /workspace

# Copy the test script
COPY test.py /workspace/test.py

# Pre-download the Hugging Face model to cache
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_name = 'microsoft/Phi-3-mini-4k-instruct'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForCausalLM.from_pretrained(model_name)"

# Set default command
CMD ["/bin/bash"]
