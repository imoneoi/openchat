FROM nvcr.io/nvidia/pytorch:23.05-py3

# install pip packages
COPY /requirements.txt /
RUN pip install -r /requirements.txt

# mouting points
RUN mkdir /ochat
RUN mkdir /model
RUN mkdir /logs

# entrypoint
EXPOSE 18888

WORKDIR /
ENTRYPOINT ["sh", "-c", "python3 -m ochat.serving.openai_api_server --model-type ${MODEL_TYPE} --model /model --host 0.0.0.0 --port 18888 --engine-use-ray --worker-use-ray --max-num-batched-tokens 5120 --disable-log-requests --disable-log-stats --log-file /logs/openchat.log"]
