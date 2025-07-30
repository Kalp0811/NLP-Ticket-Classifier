# Dockerfile

# Start with an official Python 3.10 image.
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy your application code and data into the container
COPY ./src ./src
COPY ./data/customer_support_tickets.csv ./data/customer_support_tickets.csv

# Set the Python path so imports like "src.api" work
ENV PYTHONPATH=/app

# Expose port 8000 to allow communication with the API
EXPOSE 8000

# This command runs when the container starts. It runs the entire pipeline:
# 1. Prepares the data sample
# 2. Trains the transformer model
# 3. Starts the API server
CMD ["sh", "-c", "python -m src.explore_data && python -m src.train_transformer && uvicorn src.api:app --host 0.0.0.0 --port 8000"]