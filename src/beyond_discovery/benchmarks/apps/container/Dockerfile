# The operator MUST supply a reviewed digest-pinned Python 3.11 base image.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
RUN python -m pip install --no-cache-dir numpy==1.26.4
WORKDIR /opt/apps
COPY vendor/testing_util.py /opt/apps/testing_util.py
COPY wire.py /opt/apps/wire.py
COPY container/worker.py /opt/apps/worker.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "/opt/apps/worker.py"]
