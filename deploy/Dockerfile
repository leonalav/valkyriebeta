FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Run API server
EXPOSE 8000
CMD ["python", "-m", "valkyrie.serve.api"]
