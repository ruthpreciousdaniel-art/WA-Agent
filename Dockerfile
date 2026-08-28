FROM python:3.11-slim
WORKDIR /code
COPY . .
RUN ls -la /code
CMD ["sh", "-c", "echo done"]

