FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Add missing system libraries (⬅️ THIS is the fix)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Flask env var
ENV FLASK_APP=app.py

EXPOSE 7860

CMD ["python", "app.py"]
