FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py config.example.yaml persona.yaml ./
COPY agf ./agf
COPY workflows ./workflows

VOLUME ["/app/data"]
CMD ["python", "main.py", "--log-file", "logs/bot.log"]
