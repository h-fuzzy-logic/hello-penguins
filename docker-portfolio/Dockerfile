FROM python:3.12.0-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Create mlruns directory
RUN mkdir -p /app/mlruns/ /app/mlartifacts /app/.metaflow

RUN apt-get update && apt-get install -y ca-certificates unzip nginx && rm -rf /var/lib/apt/lists/*

# Remove default Nginx config
RUN rm /etc/nginx/sites-available/default

# Remove default symbolic link
RUN rm -f /etc/nginx/sites-enabled/default

# Copy custom Nginx config
COPY nginx/default /etc/nginx/sites-available/default

# Enable the configuration
RUN ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

# Set strict permissions on sensitive files
RUN chmod 644 /etc/nginx/sites-available/default

# Extract Archive.zip
COPY /artifacts/Archive.zip /app/Archive.zip
RUN unzip /app/Archive.zip -d /app
RUN rm /app/Archive.zip && chmod -R a-w /app/mlruns

# Non-root user


EXPOSE 8080
CMD ["/bin/sh", "-c", "mlflow server --host 127.0.0.1 --port 5000 & sleep 60 && nginx -g 'daemon off;'"]

