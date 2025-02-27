FROM python:3.12

RUN mkdir /ahus
RUN mkdir /ahus/model
RUN mkdir /ahus/utils
RUN mkdir /workspace
RUN mkdir /workspace/inputs
RUN mkdir /workspace/outputs
WORKDIR /ahus
COPY src/submission/requirements.txt /ahus/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

COPY src/model/ /ahus/model/
# COPY src/utils/ /ahus/utils/
COPY src/submission/sammed_predict.py /ahus/
COPY work_dir/mar3/sam_model_latest.pth /workspace/weights.pth
COPY src/submission/predict.sh /ahus/

CMD ["sleep", "infinity"]
