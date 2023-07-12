FROM rasa/rasa-sdk:2.8.2

WORKDIR /app

COPY . /app/actions

USER root
RUN pip install --no-cache-dir -r /app/actions/requirements-actions.txt
RUN python -m spacy download en_core_web_md 
RUN python -m spacy link en_core_web_md en
RUN mkdir /.cache && chmod -R 777 /.cache

USER 1001
CMD ["start", "--actions", "actions", "--debug"]
