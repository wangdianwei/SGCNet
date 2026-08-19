# SGCNet

SGCNet: A structure-guided collaborative feature representation network for road defect detection

# Abstract

The identification of pavement defects is crucial for road upkeep and traffic safety. In complex road scenes, existing methods struggle to effectively suppress background interference within crack bounding boxes and have limited ability to extract features from road defects of different shapes and sizes. To address these challenges, we propose a structure-guided collaborative feature representation network (SGCNet) for road defect detection.
First, we propose a structure-guided crack learning (SGCL) module that uses morphological priors to generate structural pseudo labels within crack bounding boxes, providing more detailed structural supervision during training. Second, we develop a local-global adaptive fusion (LGAF) module that integrates local convolutional operations with state space modeling to jointly capture local texture features and global contextual information.
Furthermore, to improve multi-scale feature integration, we design a lightweight multi-scale selective aggregation (L-MSSA) module before the detection head. Experimental results show that SGCNet achieves mAP@0.5 values of 58.6% and 84.7% on the UAV-PDD2023 and RDD2022 China datasets, respectively, while requiring only 2.8M parameters and 6.5GFLOPs.

## Get Started

```bash
conda create -n SGCNet python=3.10 -y
conda activate SGCNet
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.3.252
pip install numpy==1.26.4
pip install mamba-ssm==1.2.0
```

### Train

```bash
python train.py
```

### Val

```bash
python val.py
```
