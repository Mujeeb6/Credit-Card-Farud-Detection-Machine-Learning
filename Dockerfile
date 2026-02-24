# 1. Use an official, lightweight Python runtime
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the dependencies file into the container
COPY requirements.txt .

# 4. Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code, templates, and dataset
COPY . .

# 6. Expose port 5000 for the web server
EXPOSE 5000

# 7. Use Gunicorn to serve the app (This shows production-readiness!)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]