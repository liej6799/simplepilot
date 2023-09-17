FROM python:3.8

WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt
COPY . .
CMD ["gunicorn", "--chdir python/", "--bind", "0.0.0.0:8000", "server:app", "-k", "gevent"]
