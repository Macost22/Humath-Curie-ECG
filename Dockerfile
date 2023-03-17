FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip3  install -r requirements.txt
COPY . .
EXPOSE 8000

CMD ["uvicorn","server:app", "--host=0.0.0.0", "--reload"]

