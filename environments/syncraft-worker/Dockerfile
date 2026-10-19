FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends autodock-vina ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 rdkit==2025.3.6 openbabel-wheel==3.1.1.22 meeko==0.7.1 gemmi==0.7.3 loguru==0.7.3 numpy==2.2.6 scipy==1.15.3 biopython==1.85 lxml==5.4.0 --no-deps
RUN pip install --no-cache-dir --no-build-isolation --no-deps plip==3.0.0
RUN apt-get -o Acquire::Retries=3 update && apt-get -o Acquire::Retries=3 install -y --no-install-recommends libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
RUN python -c "from rdkit import Chem; from openbabel import openbabel; import meeko, plip; c=openbabel.OBConversion(); assert c.SetInAndOutFormats('pdb','pdbqt'); print('SynCraft worker imports and PDB/PDBQT plugins OK')" && vina --version
WORKDIR /job
