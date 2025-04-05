#!/bin/bash
set -e

# Function for logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if model file exists, if not download it
if [ -n "$MODEL_URL" ] && [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    log "Downloading model from $MODEL_URL"
    mkdir -p ${MODEL_DIR}
    curl -L --progress-bar $MODEL_URL -o ${MODEL_DIR}/model.safetensors
    
    if [ $? -ne 0 ]; then
        log "ERROR: Failed to download model"
        exit 1
    fi
    
    log "Model downloaded successfully"
else
    log "Using existing model or no model URL provided"
fi

# Set up environment variables
export MODEL_PATH=${MODEL_PATH:-"${MODEL_DIR}/model.safetensors"}
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
export MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-100}
export REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-30}

# Wait for required services (like Redis) to be ready
if [ -n "$REDIS_URL" ]; then
    log "Waiting for Redis to be ready..."
    REDIS_HOST=$(echo $REDIS_URL | sed -E 's/redis:\/\/([^:]+).*$/\1/')
    REDIS_PORT=$(echo $REDIS_URL | sed -E 's/redis:\/\/[^:]+:([0-9]+).*$/\1/')
    REDIS_PORT=${REDIS_PORT:-6379}
    
    max_attempts=30
    attempt=0
    while ! nc -z $REDIS_HOST $REDIS_PORT; do
        if [ $attempt -ge $max_attempts ]; then
            log "ERROR: Redis not available after $max_attempts attempts"
            exit 1
        fi
        attempt=$((attempt+1))
        log "Waiting for Redis... attempt $attempt/$max_attempts"
        sleep 1
    done
    log "Redis is available"
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    log "Running database migrations..."
    python -m alembic upgrade head
    log "Migrations completed"
fi

# Setup Prometheus multiproc directory if metrics enabled
if [ "$ENABLE_METRICS" = "true" ]; then
    log "Setting up metrics collection"
    rm -rf /tmp/metrics/*
    mkdir -p /tmp/metrics
    export PROMETHEUS_MULTIPROC_DIR=/tmp/metrics
fi

# Generate an API key if none is provided
if [ -z "$API_KEY" ] && [ "$GENERATE_API_KEY" = "true" ]; then
    export API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    log "Generated API key: $API_KEY"
fi

# Check if we're in development mode
if [ "$ENVIRONMENT" = "development" ]; then
    log "Starting in development mode with hot-reload"
    exec uvicorn api.main:app --host $HOST --port $PORT --reload --log-level $LOG_LEVEL
else
    # Start the application with proper worker count
    log "Starting in production mode with $WORKERS workers"
    
    # Use different startup commands based on the worker type
    if [ "$WORKER_TYPE" = "gunicorn" ]; then
        exec gunicorn api.main:app --workers $WORKERS --worker-class uvicorn.workers.UvicornWorker \
            --bind $HOST:$PORT --log-level $LOG_LEVEL --timeout 120
    else
        # Default to uvicorn
        exec uvicorn api.main:app --host $HOST --port $PORT --workers $WORKERS --log-level $LOG_LEVEL
    fi
fi 