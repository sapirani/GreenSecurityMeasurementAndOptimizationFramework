FROM python:3.13-slim

RUN apt update
RUN apt install -y vim
RUN apt install -y dmidecode
RUN apt install -y procps
RUN apt install -y upower

WORKDIR /green_security_measurements

RUN python -m venv green_security_venv

COPY requirements requirements
RUN green_security_venv/bin/pip install -r requirements/container_requirements.txt

COPY . .

VOLUME ["/green_security_measurements"]

ENTRYPOINT ["green_security_venv/bin/python", "scanner.py"]
