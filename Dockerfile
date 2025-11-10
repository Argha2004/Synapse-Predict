# Use official Python runtime as base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the app
CMD ["gunicorn", "-b", ":9000", "app:app"]
