FROM python:3.11-slim AS builder

COPY src/juena/sandbox/repository-requirements.txt /tmp/repository-requirements.txt

# PyPI publishes an x86_64 Linux wheel for jscatter. Keep the build toolchain in
# this disposable stage so ARM64 development images can build the source
# distribution without putting compilers in the execution image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran \
    && pip install --no-cache-dir --prefix=/opt/sandbox \
        matplotlib==3.10.8 \
        numpy==2.4.2 \
        pandas==2.3.3 \
        pillow==12.1.0 \
        scikit-learn==1.8.0 \
        scipy==1.17.1 \
        seaborn==0.13.2 \
        sympy==1.14.0 \
        --requirement /tmp/repository-requirements.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash coreutils findutils grep libgfortran5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/sandbox/ /usr/local/

# Fail the image build if the repository package or its compiled extensions
# cannot be imported, then remove installers from the runtime environment.
RUN python -c "import jscatter; assert jscatter.version == '1.9.0.1'" \
    && chmod 1777 /tmp/matplotlib \
    && python -m pip uninstall -y pip setuptools wheel

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin sandbox

WORKDIR /workspace
USER sandbox

CMD ["python", "-V"]
