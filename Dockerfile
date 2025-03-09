FROM python:3.10-slim

RUN apt update
RUN apt install -y vim
RUN apt install -y dmidecode
RUN apt install -y procps
RUN apt install -y upower

WORKDIR /green_security_measurements

RUN python -m venv green_security_venv

COPY requirements requirements
RUN green_security_venv/bin/pip install -r requirements/requirements_linux.txt

COPY . .

VOLUME ["/green_security_measurements/results", "/green_security_measurements/program_parameters.py"]

#ENTRYPOINT ["green_security_venv/bin/python", "scanner.py"]
