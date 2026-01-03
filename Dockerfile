FROM python:3.9-slim-buster

# install system dependencies required for auto-sklearn
# build-essential and swig are mandatory for auto-sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app
COPY . .

# expose the port Streamlit runs on
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]

