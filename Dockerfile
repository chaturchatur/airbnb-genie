FROM python:3.9-slim-bullseye

# install system dependencies required for auto-sklearn
# build-essential and swig are mandatory for auto-sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app
COPY . .

# Expose the Flask port
EXPOSE 5000

# Generate artifacts (if needed) and start the Flask app
CMD ["sh", "-c", "python src/artifacts_manager.py && python src/app.py"]
