<p align="center">  
EH-DETR: Enhancing Two-Wheeler Helmet Detection for Small and Complex Scenes 
</p>   


In this paper, we proposes EH-DETR, a faster real-time two-wheeler helmet detection model based on the Real-Time Detection Transforme. EH-DETR incorporates a new FasterRepConvBlock structure designed using model re-parameterization techniques to improve detection performance while meeting real-time requirements. Additionally, it introduces a Mixed Global Attention module to resolve object confusion and a Cross-Stage Partial Parallel Atrous Convolution (CSPPAC) module to enhance feature fusion efficiency and receptive field size. To tackle the detection of small helmet objects, EH-DETR employs a channel-gated up-sampling and down-sampling technique. Experimental results demonstrate that EH-DETR enhances the mAP50 by 2.3% and increases the FPS to 141.3 on the helmet dataset, significantly improving the model's capability for detecting small helmets and dense scenes while ensuring real-time performance. 

# **Overview**

![EH-DETR](https://github.com/user-attachments/assets/dfd0af52-aa2a-484a-b51a-edbefac56ae0)

# **Installation**

Clone repo and install requirements.txt in a Python>=3.8.0 conda environment, including PyTorch>=1.12.
```
git clone https://github.com/Eason215xB/EH-DETR.git
cd /ultralytics
pip install -r requirements.txt
```

# **Usage**

Dataset: [helmet-dataset](https://pan.baidu.com/s/17Jpwt5Nhz1x8gUgbUu3XPA?pwd=92rx) 

Password:  92rx

# **Training**

```
cd /ultralytics
python train.py
```
## Reference

- If you found our work useful in your research, please consider citing our works at:

```tex

@article{10.1117/1.JEI.34.1.013035,
author = {Liwei Liu and Xinbo Yue and Ming Lu and Pingge He},
title = {{EH-DETR: enhanced two-wheeler helmet detection transformer for small and complex scenes}},
volume = {34},
journal = {Journal of Electronic Imaging},
number = {1},
publisher = {SPIE},
pages = {013035},
keywords = {RT-DETR, object detection, helmet detection, transformer target detection},
year = {2025},
doi = {10.1117/1.JEI.34.1.013035},
URL = {https://doi.org/10.1117/1.JEI.34.1.013035}
}
