FROM python:3.12

RUN mkdir /ahus
RUN mkdir /ahus/src
RUN mkdir /workspace
RUN mkdir /workspace/inputs
RUN mkdir /workspace/outputs
WORKDIR /ahus
COPY src/submission/requirements.txt /ahus/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

COPY src/ /ahus/src/
COPY src/submission/ahus_predict.py /ahus/
COPY submission_files/liere_2_may.pth /workspace/weights.pth
COPY src/submission/predict.sh /ahus/

CMD ["sleep", "infinity"]
