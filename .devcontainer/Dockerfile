FROM mcr.microsoft.com/devcontainers/python:3.10

RUN python -m pip install --no-cache-dir --upgrade pip poetry \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 'torch>=2.0.0' \
    && pip install --no-cache-dir --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html torch_geometric pyg_lib torch_scatter torch_sparse

ENV POETRY_VIRTUALENVS_CREATE=false
COPY pyproject.toml poetry.lock* /tmp/poetry/
RUN poetry -C /tmp/poetry --no-cache install --no-root --no-directory