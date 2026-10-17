FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
# Install the CPU-only torch build first; the default Linux wheel bundles GBs of unused CUDA libraries.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
