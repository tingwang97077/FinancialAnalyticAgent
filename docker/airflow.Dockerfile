FROM apache/airflow:3.3.0-python3.12

ARG AIRFLOW_VERSION=3.3.0

USER airflow
WORKDIR /opt/airflow/ffa

COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    "beautifulsoup4>=4.13,<5" \
    "httpx>=0.28,<1" \
    "langfuse>=3,<4" \
    "lxml>=6,<7" \
    "psycopg[binary,pool]>=3.2,<4" \
    "pydantic-settings>=2.10,<3" \
    "sqlalchemy>=2,<2.1" \
    "tenacity>=9,<10" \
    "tiktoken>=0.9,<1" \
    && pip install --no-cache-dir --no-deps .

WORKDIR /opt/airflow
