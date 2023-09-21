# Training ReID model on MOT format

## Setup
```
pip install -r requirements.txt
```
### Setup environment path
In the [.env] file, rename the variables as following:
- *DATASETS.ROOT_DIR*: the path to the dataset directory
- *PRETRAIN_ROOT*: the path to the pretrain directory 

#### Data structure:
Please format the reid train from cropped images from MOT dataset . Then put it under the path like the following structure:
```
data/
├── person_reid/
│   ├── gallery/
│   ├── query/
│   └── train/
└──
```
## Pretrained on ImageNet
Put the pretrained model like the following structure
```
├── datasets/
│   ├── reid/
│   ├── pretrain/
│       ├── resnet50-19c8e357.pth
│       ├── -------
│       └── jx_vit_base_p16_224-80ecf9dd.pth

```

## Training ReID
```
bash run.sh
```

Before training create `logs/` folder to save status from every running state .After training, the weight will be stored in the `lightning_logs/` folder. Navigate to this folder and copy the corresponding epoch weight of each model to the corresponding folder in `output/weight`.
