FROM tensorflow/tensorflow:2.18.0 AS build
WORKDIR /app
COPY . /app
RUN apt-get remove python3-blinker --yes
RUN pip install --no-cache-dir -r install/requirements.txt

VOLUME /app/results

EXPOSE 9000

ENTRYPOINT python /app/flask_app/app.py
