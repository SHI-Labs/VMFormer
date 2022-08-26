## Installation

Step 1: Clone this repo
```bash
git clone https://github.com/SHI-Labs/VMFormer
```

Step 2: Create conda environment
```bash
conda create --name vmformer python=3.7
conda activate vmformer
```

Step 3: Install pytorch and torchvision

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
```

Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Step 5: Install CUDA version of MultiScaleDeformableAttention

```bash
cd ./models/ops
sh ./make.sh
python test.py
```


