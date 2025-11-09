## Toward Realistic Camouflaged Object Detection: Benchmarks and Method


<div align="center"><img src="RCOD.png" width="500"></div>

## Datasets with BBox and Classes Link (First Version) 
[Google](https://drive.google.com/drive/folders/1SafBDHRbutQ4D3yqDPOEmZY2u9Ip7Feh) 

[Baidu](https://pan.baidu.com/s/11m8pSerp4hR6pMZMD7WiyQ?pwd=93yd)  Extract Code: 93yd 
   
The first version of the datasets contain category and bounding box annotations.

| Datasets | Categories | Training Images | Test Images |
| ---- | ---- | ---- | ---- |
| COD10K-D | 68 | 6000 | 4000 |
| NC4K-D | 37 | 2863 | 1227 |
| CAMO-D | 43 | 744 | 497 |

## Benchmarks on COD10K-D, NC4K-D, and CAMO-D datasets (First Version) 

## Performance Comparison on COD10K-D, NC4K-D, and CAMO-D Datasets

**Caption**: Performance of various detection methods on COD10K-D, NC4K-D, and CAMO-D datasets. The results for a single seed are presented here. The paper presents 10 random seed results.

### Generic Methods
| Method | Backbone | mAP | AP50 | AP75 | APm | APl | mAP | AP50 | AP75 | APm | APl | mAP | AP50 | AP75 | APm | APl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | **COD10K-D** | | | | | **NC4K-D** | | | | | **CAMO-D** | | | | |
| YOLOv7-L | CSPDarknet | 3.8 | 8.2 | 2.8 | 1.0 | 4.0 | 6.8 | 14.2 | 6.0 | 1.7 | 7.3 | 5.4 | 10.2 | 5.5 | 8.5 | 5.4 |
| Faster R-CNN | ResNet-50 | 8.3 | 21.0 | 5.0 | 4.8 | 8.8 | 19.2 | 39.7 | 16.0 | 8.1 | 20.2 | 4.7 | 12.4 | 2.2 | 3.5 | 5.3 |
| YOLOv8-L | CSPVoVNet | 9.7 | 16.8 | 9.4 | 2.6 | 10.4 | 23.5 | 34.9 | 25.2 | 10.6 | 24.7 | 25.4 | 37.2 | 26.1 | 14.4 | 26.7 |
| Faster R-CNN | ResNet-101 | 10.8 | 24.4 | 7.7 | 9.2 | 11.6 | 23.0 | 47.2 | 20.1 | 10.4 | 24.0 | 9.3 | 21.1 | 6.9 | 9.9 | 10.1 |
| Def-DETR | ResNet-50 | 12.2 | 23.1 | 11.4 | 6.5 | 13.1 | 27.4 | 49.6 | 27.9 | 14.0 | 29.7 | 13.3 | 26.9 | 12.4 | 9.7 | 14.1 |
| Def-DETR | ResNet-101 | 13.5 | 23.7 | 13.5 | 9.2 | 14.6 | 30.9 | 54.4 | 32.0 | 12.4 | 32.5 | 13.7 | 27.4 | 12.7 | 13.2 | 15.5 |
| Cascade R-CNN | ResNet-101 | 15.3 | 27.4 | 15.9 | 8.5 | 16.4 | 27.5 | 46.9 | 28.9 | 11.3 | 29.2 | 14.0 | 26.6 | 13.1 | 13.1 | 14.8 |
| Faster R-CNN | Swin-T | 16.3 | 35.3 | 13.1 | 8.6 | 17.4 | 29.1 | 58.8 | 25.6 | 16.8 | 30.4 | 11.3 | 32.3 | 5.5 | 8.6 | 12.0 |
| Faster R-CNN | Swin-L | 32.1 | 54.6 | 33.1 | 17.1 | 33.9 | 49.1 | 75.8 | 55.1 | 22.7 | 51.3 | 34.2 | 67.4 | 30.2 | 24.0 | 36.1 |

### Large Vision-Language Models
| Method | Backbone | mAP | AP50 | AP75 | APm | APl | mAP | AP50 | AP75 | APm | APl | mAP | AP50 | AP75 | APm | APl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | **COD10K-D** | | | | | **NC4K-D** | | | | | **CAMO-D** | | | | |
| GLIP | Swin-T | 26.4 | 36.3 | 28.5 | 14.7 | 28.0 | 49.6 | 63.7 | 53.4 | 23.9 | 51.6 | 32.6 | 42.9 | 33.6 | 40.9 | 35.1 |
| **GLIP + CAFR** | **Swin-T** | **28.8** | **38.2** | **31.0** | **16.4** | **30.6** | **53.3** | **67.7** | **55.2** | **27.0** | **55.7** | **34.7** | **43.5** | **34.3** | **36.9** | **38.5** |
| GLIP | Swin-L | 40.2 | 47.9 | 43.5 | 24.7 | 42.3 | 76.9 | 86.9 | 80.9 | 50.4 | 78.5 | 63.0 | 74.4 | 68.1 | 52.4 | 66.8 |
| **GLIP + CAFR** | **Swin-L** | **42.9** | **50.0** | **45.8** | **28.3** | **43.0** | **78.7** | **89.9** | **83.9** | **53.5** | **80.5** | **63.6** | **77.3** | **70.5** | **50.0** | **69.8** |
| GDino | Swin-T | 44.8 | 56.0 | 47.9 | 23.5 | 47.8 | 69.8 | 81.0 | 72.1 | 37.5 | 72.4 | 48.0 | 59.1 | 52.4 | 40.7 | 52.2 |
| **GDino + CAFR** | **Swin-T** | **48.5** | **60.7** | **51.9** | **28.7** | **49.7** | **72.3** | **82.7** | **74.5** | **35.6** | **74.7** | **50.7** | **60.7** | **55.3** | **45.0** | **55.2** |
| GDino | Swin-B | 58.7 | 70.9 | 63.1 | 23.6 | 62.3 | 79.9 | 90.5 | 84.6 | 54.8 | 81.5 | 68.6 | 80.6 | 75.1 | 55.7 | 73.0 |
| **GDino + CAFR** | **Swin-B** | **62.3** | **74.8** | **68.3** | **35.4** | **67.1** | **81.9** | **92.9** | **87.3** | **58.1** | **83.7** | **72.1** | **83.3** | **77.2** | **56.9** | **74.5** |

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


