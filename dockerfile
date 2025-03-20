FROM python:3.9-slim

# Install system packages if needed
RUN apt-get update && apt-get install -y gcc g++ make

WORKDIR /app

# Copy the requirements file into the container
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install -r requirements-api.txt

# Copy script
COPY script.py /app/script.py

EXPOSE 8080

CMD ["python", "script.py"]
