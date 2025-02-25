FROM python:3.12

RUN mkdir /cvpr
RUN mkdir /workspace
RUN mkdir /workspace/inputs
RUN mkdir /workspace/outputs
WORKDIR /cvpr
COPY submission_files/requirements.txt /cvpr/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

COPY submission_files/* /cvpr/

CMD ["sleep", "infinity"]
