# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY face_extractor.py . 
COPY service.py .
COPY cluster_ids.pkl .
COPY cluster_mediods.pkl .  

# Expose the port that the FastAPI service will listen on
EXPOSE 8000

# Define the command to run the FastAPI service
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
