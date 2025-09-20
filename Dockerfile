FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
ENV PORT=8080
EXPOSE $PORT
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8080"]
