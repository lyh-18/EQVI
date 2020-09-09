# EQVI-Enhanced Quadratic Video Interpolation
## winner solution of AIM2020 VTSR Challenge
Authors: Yihao Liu, Liangbin Xie, Li Siyao, Wenxiu Sun, Yu Qiao, Chao Dong
[paper]

If you find our work is useful, please kindly cite it.
```
@InProceedings{liu2020enhanced,  
author = {Yihao Liu and Liangbin Xie and Li Siyao and Wenxiu Sun and Yu Qiao and Chao Dong},  
title = {Enhanced quadratic video interpolation},  
booktitle = {European Conference on Computer Vision Workshops},  
year = {2020},  
}
```
## Preparation
### Dependencies
- Python >= 3.6
- Tested on PyTorch==1.2.0 (may work for other versions)
- Tested on Ubuntu 18.04.1 LTS
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Install correlation package
In our implementation, we use [ScopeFlow](https://github.com/avirambh/ScopeFlow) as a pretrained flow estimation module.  
Please follow the instructions to install the required correlation package:
```
cd models/scopeflow_models/correlation_package
python setup.py install
```
Note:
if you use CUDA>=9.0, just execute the above commands straightforward;  
if you use CUDA==8.0, you need to change the folder name 'correlation_package_init' into 'correlation_package', and then execute the above commands.

Please refer to [ScopeFlow](https://github.com/avirambh/ScopeFlow) and [irr](https://github.com/visinf/irr) for more information.

### Download pretrained models
The pretrained models can be downloaded at [Google Driver]().  
Unzip it in the root dir.
