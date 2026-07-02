FROM ubuntu:20.04

# Install Python 3.8 and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Link python3.8 to python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Add this line before installing open3d
RUN apt-get update && apt-get install -y libgomp1 && apt-get install -y libgl1-mesa-glx

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set default command
CMD ["python", "main.py"]
