
# Use Python 3.11
FROM python:3.11-slim

# Install the pinned uv version
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

# Python and uv configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set the application directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy application source code and knowledge corpus
COPY src ./src
COPY corpus ./corpus

# Install production dependencies using the lockfile
RUN uv sync --frozen --no-dev

# Expose the FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uv", "run", "--no-sync", "uvicorn", "pharma_research_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
