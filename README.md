# SASNet (AAAI2021)

 Official implementation in PyTorch of **SASNet** as described in "To Choose or to Fuse? Scale Selection for Crowd Counting" by Qingyu Song *, Changan Wang *, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Jian Wu, Jiayi Ma.

<p align="center"> <img src="imgs/title.png" width="80%" />

The codes is tested with PyTorch 1.5.0. It may not run with other versions.
 
## Visualizations for the scale-adaptive selection
The proposed adaptive selection strategy automatically learns the internal relations and the following visualizations demonstrate its effectiveness.

<p align="center"><img src="imgs/fig1.png" width="80%"/>

## Installation
* Clone this repo into a directory named SASNet_ROOT
* Install Python dependencies. We use python 3.6.8 and pytorch >= 1.5.0
```
pip install -r requirements.txt
```
* Download ShanghaiTech dataset and models from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)

## Preparation
Organizing the datas and models as following:
```
SASNet_ROOT/
        |->datas/
        |    |->part_A_final/
        |    |->part_B_final/
        |    |->...
        |->models/
        |    |->SHHA.pth
        |    |->SHHB.pth
        |    |->...
        |->main.py
```
Generating the density maps for the data:
```
python prepare_dataset.py --data_path ./datas/part_A_final
python prepare_dataset.py --data_path ./datas/part_B_final
```

## Running

Run the following commands to launch inference:

```
python3 main.py --data_path ./datas/part_A_final --model_path ./models/SHHA.pth 
python3 main.py --data_path ./datas/part_B_final --model_path ./models/SHHB.pth 
```

## The network
The overall architecture of the proposed SASNet mainly consists of three components: U-shape backbone, confidence branch and density branch.

<img src="imgs/main.png"/>

## Comparison with state-of-the-art methods
The SASNet achieved state-of-the-art performance on several challenging datasets with various densities.

<img src="imgs/results.png"/>

## Qualitative results
The following qualitative results show impressive counting accuracy under various crowd densities.

<img src="imgs/vis.png"/>


## Citing SASNet

If you think SASNet is useful in your project, please consider citing us.

```BibTeX
@article{sasnet,
  title={To Choose or to Fuse? Scale Selection for Crowd Counting},
  author={Qingyu Song and Changan Wang and Yabiao Wang and Ying Tai and Chengjie Wang and Jilin Li and Jian Wu and Jiayi Ma},
  journal={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021}
}
```