# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     some-package && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . /app

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]

# docker build -t insurance .
# docker run -p 8080:8080 insurance
# docker tag insurance maheshp23/insurance:latest
# docker push maheshp23/insurance
# docker pull maheshp23/insurance