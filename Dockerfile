FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY configs/ configs/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
