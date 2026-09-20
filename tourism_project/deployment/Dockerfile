# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the deployment requirements file into the container
COPY tourism_project/deployment/requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application and the trained model into the container
COPY tourism_project/deployment/app.py .
COPY tourism_project/deployment/xgb_model.joblib .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=", "0.0.0.0", "--server.enableXsrfProtection=false"]
