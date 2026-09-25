# CUDA-Q quantum simulator image for 高火力 DOK.
# libgfortran5 is needed by cudaq-solvers (not bundled in its wheel).
# The cudaq wheel pulls the CUDA runtime libraries from PyPI; the GPU driver comes from the DOK host.
# Debian 12 (bookworm), not the Debian 13 default: libcudaq-solvers.so requests an executable
# stack, which glibc 2.41+ refuses to load ("cannot enable executable stack").
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends procps libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run.sh /app/run.sh
COPY bench /app/bench
COPY vqe /app/vqe
RUN chmod +x /app/run.sh

WORKDIR /app
ENTRYPOINT ["/app/run.sh"]
CMD ["all"]
