FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x start_backend.sh

EXPOSE 5001

CMD ["./start_backend.sh"]