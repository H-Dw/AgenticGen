FROM mambaorg/micromamba:2.3.2
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes
ENV PATH=/opt/conda/bin:$PATH
RUN python -c "import torch, rdchiral, rdkit, omegaconf, networkx, graphviz; print('SimpRetro worker imports OK')" && micromamba list --explicit > /opt/conda/agenticgen-environment.lock.txt
WORKDIR /job
