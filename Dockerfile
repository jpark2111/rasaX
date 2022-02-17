FROM rasa/rasa:2.8.2

WORKDIR /app
COPY . /app
USER root

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md 
RUN python -m spacy link en_core_web_md en

COPY ./data /app/data
COPY config.yml /app
COPY endpoints.yml /app

RUN rasa train

VOLUME /app
VOLUME /app/data
VOLUME /app/models

CMD ["run", "-m", "/app/models", "--enable-api","--cors","*","--debug" ,"--endpoints", "endpoints.yml", "--log-file", "out.log", "--debug"]

EXPOSE 5005