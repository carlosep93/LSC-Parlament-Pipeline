#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install Cython
pip3 install -r requirements2.txt

sudo apt-get update -y && sudo apt-get install -y libsndfile1 ffmpeg pdftohtml
pip3 install nemo_toolkit['all']
pip3 install apertium

# Install apertium 
# s'utilitza nomes per traduir nombres 23 -> twenty-three -> vint-i-tres
curl -sS https://apertium.projectjj.com/apt/install-release.sh | sudo bash
sudo apt-get -f install apertium-all-dev
sudo apt-get install apertium apertium-all-dev
sudo apt-get install apertium-cat-spa
sudo apt-get install apertium-cat-eng
sudo apt-get install apertium-spa-eng

git clone https://github.com/NVIDIA/NeMo.git

pre-commit install

