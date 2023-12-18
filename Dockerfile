FROM python:3.10

WORKDIR /usr/d2m

RUN mkdir -p assets/data/raw

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install dvc
RUN pip3 install flask flask-restful

COPY src ./src
COPY dvc.yaml ./dvc.yaml
COPY params_default.yaml ./params.yaml
COPY params_default.yaml ./params_default.yaml

EXPOSE 5000

RUN git init
RUN dvc init --no-scm

CMD ["python3", "src/api.py"]
