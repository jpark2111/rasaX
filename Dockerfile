FROM rasa/rasa-sdk:2.8.2

WORKDIR /app

COPY actions /app/actions

USER root
RUN pip install --no-cache-dir -r /app/actions/requirements-actions.txt

USER 1001
CMD ["start", "--actions", "actions"]