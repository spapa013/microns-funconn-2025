FROM ubuntu:24.04
ENV DEBIAN_FRONTEND noninteractive
LABEL mantainer="Zhuokun Ding <zkding@outlook.com>"
# The following dockerfile is based on jupyter/docker-stacks: https://github.com/jupyter/docker-stacks

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.12
# # Add profiling library support
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    curl \
    # - bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    ca-certificates \
    locales \
    sudo \
    # - tini is installed as a helpful container entrypoint that reaps zombie
    #   processes and such of the actual executable we want to start, see
    #   https://github.com/krallin/tini#why-tini for details.
    tini \
    # - pandoc is used to convert notebooks to html files
    #   it's not present in aarch64 ubuntu image, so we install it here
    pandoc \
    pkg-config \
    wget \
    # Common useful utilities
    nano-tiny \
    tzdata \
    unzip \
    vim-tiny \
    # git-over-ssh
    openssh-client \
    # less is needed to run help in R
    # see: https://github.com/jupyter/docker-stacks/issues/1588
    less \
    # nbconvert dependencies
    # https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    # Enable clipboard on Linux host systems
    xclip \
    # R pre-requisites
    fonts-dejavu \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV HOME="/"

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
   # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
   echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc


# Download and install Micromamba, and initialize Conda prefix.
#   <https://github.com/mamba-org/mamba#micromamba>
#   Similar projects using Micromamba:
#     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
#     - repo2docker: <https://github.com/jupyterhub/repo2docker>
# Install Python, Mamba and jupyter_core
# Cleanup temporary files and remove Micromamba
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
COPY initial-condarc "${CONDA_DIR}/.condarc"
WORKDIR /tmp
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba && \
    PYTHON_SPECIFIER="python=${PYTHON_VERSION}" && \
    # Bootstrap mamba and install conda available packages
    bin/micromamba install \
        --root-prefix="${CONDA_DIR}" \
        --prefix="${CONDA_DIR}" \
        --yes \
        "${PYTHON_SPECIFIER}" \
        'mamba' &&\
    rm bin/micromamba
# Pin major.minor version of python
RUN mamba list python | grep 'python ' | sed 's/^[ \t]*//' | tr -s ' ' | cut -d ' ' -f 1,2 >> "${CONDA_DIR}/conda-meta/pinned"
RUN mamba install --yes \
        'git' \
        'pip' \
        'notebook' \
        'jupyterlab' \
        'jupyterhub' \
        'jupyterlab-lsp' \
        'python-lsp-server' \
        'r-languageserver' \
        'numpy' \
        'libblas=*=*mkl'\
        'cython' \
        'dask' \
        'dill' \
        'h5py' \
        'ipympl'\
        'ipywidgets' \
        'matplotlib-base' \
        'numexpr' \
        'openpyxl' \
        'pandas' \
        'patsy' \
        'protobuf' \
        'scikit-image' \
        'scikit-learn' \
        'scipy' \
        'seaborn' \
        'sqlalchemy' \
        'statsmodels' \
        'widgetsnbextension'\
        'r-arrow' \
        'r-base' \
        'r-broom.mixed' \
        'r-caret' \
        'r-crayon' \
        'r-devtools' \
        'r-dharma' \
        'r-e1071' \
        'r-emmeans' \
        'r-forecast' \
        'r-hexbin' \
        'r-htmltools' \
        'r-htmlwidgets' \
        'r-irkernel' \
        'r-nycflights13' \
        'r-randomforest' \
        'r-rcurl' \
        'r-rmarkdown' \
        'r-rodbc' \
        'r-rsqlite' \
        'r-shiny' \
        'r-tidyverse' \
        'r-tidymodels' \
        'r-glmmTMB' \
        'r-lme4' \
        'r-performance' \
        'rpy2' \
        'unixodbc' \
        'holoviews' \
        'bokeh' \
        'panel' \
        'hvplot' \
        'datashader' \
        'param' \
        'colorcet' \
        'pytest' \
        'pytest-cov' \
        'simplejson' \
        'networkx' \
        'pylint' \
        'tqdm'

# install r packages from source due to interdependencies between glmmTMB/TMB/Matrix
# for more, see: https://glmmtmb.github.io/glmmTMB/#:~:text=glmmTMB/TMB/Matrix%20mismatches
RUN R -e "install.packages('Matrix', repos='https://cloud.r-project.org/')" && \
    R -e "install.packages('TMB', type='source', repos='https://cloud.r-project.org/')" && \
    R -e "install.packages('glmmTMB', type='source', repos='https://cloud.r-project.org/')"

RUN mamba update ffmpeg
RUN jupyter notebook --generate-config && \
    mamba clean --all -f -y && \
    npm cache clean --force && \
    jupyter lab clean


# Currently need to have both jupyter_notebook_config and jupyter_server_config to support classic and lab
COPY docker/jupyter_server_config.py /etc/jupyter/
# Add R mimetype option to specify how the plot returns from R to the browser
COPY docker/Rprofile.site /opt/conda/lib/R/etc/

# Legacy for Jupyter Notebook Server, see: [#1205](https://github.com/jupyter/docker-stacks/issues/1205)
RUN sed -re "s/c.ServerApp/c.NotebookApp/g" \
    /etc/jupyter/jupyter_server_config.py > /etc/jupyter/jupyter_notebook_config.py

# Install fonts
#! /bin/bash
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
    apt update && \
    apt install msttcorefonts -qy && \
    rm ~/.cache/matplotlib -rf

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="${HOME}/.cache/"
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"

# Download data files to /data from s3://bossdb-open-data/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/
RUN mkdir -p /data && \
    cd /data && \
    wget https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/node_data_v1.pkl && \
    wget https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/edge_data_v1.pkl

# Install funconn
COPY --chown=root:root . /src/microns-funconn-2025/
RUN cd /src/microns-funconn-2025 && pip install -e .

WORKDIR "/src/microns-funconn-2025"
