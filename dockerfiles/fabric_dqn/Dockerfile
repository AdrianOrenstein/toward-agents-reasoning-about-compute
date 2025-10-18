FROM python:3.13

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install \
    torch torchrl tensordict opt_einsum lightning lightning-bolts \
    ale-py opencv-python-headless lz4 \
    ruff pre-commit \
    tyro tqdm loguru \
    numpy wandb tensorboard pygame moviepy imageio

RUN git config --global --add safe.directory /app

ENV PYTHONPATH="/app/"

RUN useradd -u 1000 -m ubuntu
WORKDIR /app
USER ubuntu

CMD [ "/bin/bash" ]