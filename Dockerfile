# Use a lightweight official Python image
FROM python:3.10-slim

# Avoid Python buffering so logs show up immediately
ENV PYTHONUNBUFFERED=1

# Install system dependencies for image processing + HF
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Run your app
CMD ["python", "app.py"]
