# HRNet-SynthFace
This is a facial landmark detector trained on the SynthFace dataset, a subset of Microsoft's [SynthMoCap](https://github.com/microsoft/SynthMoCap) dataset.

The code for the [HRNet](https://github.com/HRNet/HRNet-Image-Classification) part was modified from the original [High-resolution networks (HRNets) for Image classification](https://github.com/HRNet/HRNet-Image-Classification) repository.

The file `yolov11n-face.pt` was downloaded from [yolo-face](https://github.com/akanametov/yolo-face).

The file `Creating an Actor-specific Facial Rig from Performance Capture - 2947688.2947693.pdf.png` is an image taken from Fig. 3 of the paper [Creating an Actor-specific Facial Rig from Performance Capture](https://dl.acm.org/doi/10.1145/2947688.2947693).

Please note that the SynthMoCap dataset, YOLO file, and the image mentioned above are subject to their own licenses and **not** the MIT license of this repository.

# Requirements
To run the code, the following packages are required:
```torch```, ```torchvision```, ```opencv (cv2)```, ```PyYAML```, ```yacs```, ```safetensors```, and ```ultralytics``` (Test with `pred_roi.py`).

For training, you will also need:
```accelerate``` and ```tensorboard```.

# Train
    accelerate launch --config_file=your_conf.conf main_softargmax_roi.py

# Prediction
    python pred_roi.py
    # or
    python pred_roi_cam.py

# Results
Below are the training results.

## Loss
![loss](images/train_loss.jpg)

## Ground Truth
![GT](images/gt.png)

## Prediction
![Pred](images/pred.png)

## Test
![Testimg](images/test.jpg)
