FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# On met tout le dossier "service/" dans /app/service (conserve le package)
WORKDIR /app
COPY . /app/service

# Installe les deps de l’API
RUN pip install --no-cache-dir -r /app/service/requirements.txt

# Assure que Python voit /app pour "import service"
ENV PYTHONPATH=/app

EXPOSE 8080
# Lance uvicorn avec le module package "service.app:app" (et plus "app:app")
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8080"]
