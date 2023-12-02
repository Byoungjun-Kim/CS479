```bash
conda create --name diffae python=3.8
conda activate nerf-tutorial
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchmetrics[image]
pip install tensorboard

export PYTHONPATH=.

conda install matplotlib
conda install scikit-learn
conda install tqdm
conda install pyyaml
```