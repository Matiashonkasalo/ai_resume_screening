FROM apache/airflow:2.8.1

# Copy requirements as root (file permissions)
USER root
COPY requirements.txt /requirements.txt

# Switch to airflow BEFORE pip install
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

# Just in case copy src
COPY src /opt/airflow/src
