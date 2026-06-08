# Start from an official Python 3.11 image
# This is the base layer — a minimal Linux OS with Python pre-installed
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Copy requirements.txt first and install dependencies
# Docker caches this layer — if requirements don't change,
# it skips reinstalling on every rebuild (saves time)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app folder containing main.py and model artifacts
COPY app/ ./app/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
# This is identical to how we started uvicorn in Colab
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]