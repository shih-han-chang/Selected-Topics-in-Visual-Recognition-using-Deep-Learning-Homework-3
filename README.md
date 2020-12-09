# Selected-Topics-in-Visual-Recognition-using-Deep-Learning-Homework-3
The target of homework 3 is Instance segmentation for Tiny PASCAL VOC dataset and the dataset contains only 1349 train images, 100 test images with 20 common object classes.

## Hardware   
  Use Linux with PyTorch to train this model  
  - PyTorch >= 1.1.  
  - Python >= 3.6.  
  - tensorboardX  
  - Other common packages.  

## Dataset 
  * The training data have 1349 images with a target file in COCO annotations format.
  * The test data have 100 images. 
  
## Training model
  * First download the weights from here https://drive.google.com/file/d/1vaDqYNB__jTB7_p9G6QTMvoMDlGkHzhP/view
  * To train YOLACT using the train script simply specify the parameters listed in train.py as a flag or manually change them.  
    - python train.py --config=res101_custom_config
## Evalution
  * To evaluate a trained network:  
    - python eval.py --trained_model=res101_custom_100000.pth
  * Then, it will generate a.json file with test result
