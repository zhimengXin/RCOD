## Toward Realistic Camouflaged Object Detection: Benchmarks and Method


<div align="center"><img src="dataset.png" width="500"></div>

## Datasets with BBox and Classes Link (First Version) 
[Google](https://drive.google.com/drive/folders/1SafBDHRbutQ4D3yqDPOEmZY2u9Ip7Feh) 

[Baidu](https://pan.baidu.com/s/11m8pSerp4hR6pMZMD7WiyQ?pwd=93yd)  Extract Code: 93yd 
   

| Datasets | Categories | Training Images | Test Images |
| ---- | ---- | ---- | ---- |
| COD10K-D | 68 | 6000 | 4000 |
| NC4K-D | 37 | 2863 | 1227 |
| CAMO-D | 43 | 744 | 497 |

## Datasets with BBox, Classes and Languages Link (Second Version) 

New datasets will be provided after the paper is published. 

**Legend:**
- B-Boxes: Bounding Boxes
- Pos-Classes: positive classes
- Tr-Samples: training samples
- Te-Samples: Test Samples

| Dataset Type | Datasets | Category | Box | Description | B-Boxes | Pos-Classes | Languages | Tr-Samples | Te-Samples |
|--------------|----------|----------|-----|-------------|---------|-------------|-----------|------------|------------|
| COD | CHAMELEON  |  |  |  | - | - | - | - | 76 |
| COD | CAMO  | ✓ |  |  | - | 8 | - | 1,000 | 250 |
| COD | NC4K  |  |  |  | - | - | - | - | 4,121 |
| COD | COD10K | ✓ | ✓ |  | 5,899 | 69 | - | 6,000 | 4,000 |
| **RCOD** | COD10K-D | ✓ | ✓ | ✓ | 11,684 | 81 | 10,798 | 6,172 | 5,734 |
| **RCOD** | RCOD-D | ✓ | ✓ | ✓ | 12,955 | 59 | 11,850 | 4,192 | 5,846 |



## Framework install

<div align="center"><img src="finetuning.png" width="1000"></div>

Our code is based on MMDetection. Here, for the convenience of readers, we have uploaded the full code of mmdetection and our code. If the relevant environment for mmdetection is configured on your server, you can download and use it directly. MMDetection is an open source object detection toolbox based on PyTorch. We adopt MMDetection as our baseline framework from [MMdetection](https://github.com/open-mmlab/mmdetection)


**Our environmental installation**
* Linux with Python >= 3.10
* conda create -n RCOD python==3.10
* conda activate RCOD
* [PyTorch](https://pytorch.org/get-started/locally/) >= 2.1.1 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version.
* Our CUDA is 11.8
* Install PyTorch 2.1.1 with CUDA 11.8 
  ```shell
  conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
* pip install mmcv>=2.2.0
* pip install -r requirements/build.txt
* pip install -v -e . 

**Training on CAFR**

* We provide the config files of the three datasets together, thus the number of categories in the config file and the path of the dataset needed to be changed during training. Here, data modification includes:
```
  RCOD/mmdet/datasets/coco.py  
  RCOD/configs/_base_/coco_detection.py
```

* We use GLIP+APG as an example to show the training processing:
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh "--config configs/glip/glip_swin_tiny_cafr.py --work-dir /home/output 4
  ```


## Citation

If you use this toolbox or benchmark datasets in your research, please cite this project.

```
@article{rcod,
	title={Toward Realistic Camouflaged Object Detection: Benchmarks and Method},
	author={Xin, Zhimeng and Wu, Tianxu and Chen, Shiming and Ye, Shuo and Xie, Zijing and Zou, Yixiong and You, Xinge and Guo, Yufei},
	journal={arXiv preprint arXiv:2501.07297},
	year={2025}
}
```

We need your assistance.. Can you help?
kindly respond to this email...meryemshah26@gmail.com
