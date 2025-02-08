FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm AS base

ENV PYTHONPATH=/code

RUN pip install --no-cache "poetry>=1.8,<1.9" 
RUN poetry config virtualenvs.create false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        libgdal-dev \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --only main --no-interaction --no-ansi --no-root -vv \
    && rm -rf /root/.cache/pypoetry

FROM base AS fastapi

RUN adduser --disabled-password --gecos '' fastapi

RUN poetry install -E fastapi --only main --no-interaction --no-ansi --no-root -vv \
    && rm -rf /root/.cache/pypoetry

COPY ./thisapp ./thisapp

RUN poetry install -E fastapi --only main --no-interaction --no-ansi -vv \
    && rm -rf /root/.cache/pypoetry

COPY ./fastapi-app ./

USER fastapi
EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS streamlit

RUN adduser --disabled-password --gecos '' streamlit

RUN poetry install -E streamlit --only main --no-interaction --no-ansi --no-root -vv \
    && rm -rf /root/.cache/pypoetry

COPY ./thisapp ./thisapp

RUN poetry install -E streamlit --only main --no-interaction --no-ansi -vv \
    && rm -rf /root/.cache/pypoetry

COPY ./streamlit-app ./

USER streamlit
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app/Home.py"]

FROM fastapi AS devcontainer

USER root

RUN apt-get update \
    && apt-get install -y \
        curl \
        docker.io \
        git \
        htop \
        procps \
        time \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean 

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install --update \
    && echo 'complete -C '/usr/local/bin/aws_completer' aws' >> ~/.bashrc \
    && rm -rf awscliv2.zip ./aws

RUN poetry install --all-extras --no-interaction --no-ansi --no-root -vv \
    && rm -rf /root/.cache/pypoetry

COPY ./thisapp ./thisapp

RUN poetry install --all-extras --no-interaction --no-ansi -vv \
    && rm -rf /root/.cache/pypoetry

WORKDIR /workspace

ENTRYPOINT ["tail", "-f", "/dev/null"]
