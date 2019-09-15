# ---- Base ----
# ---- Python ----
FROM python:3 AS build
MAINTAINER "DeltaML dev@deltaml.com"
COPY requirements.txt .
# install app dependencies
RUN pip install  --user -r requirements.txt

FROM python:stretch AS release
WORKDIR /app
COPY --from=build /root/.local /root/.local
ADD /federated_aggregator /app/federated_aggregator
ENV PATH=/root/.local/bin:$PAT
ENV ENV_PROD=1
EXPOSE 8080
CMD [ "gunicorn", "-b", "0.0.0.0:8080", "wsgi:app", "--chdir", "federated_aggregator/", "--preload"]
