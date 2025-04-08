from safetensors.torch import load_file
from config import config
from config import update_config
from dataset import SynthFaceDataset_RoI
import heatmap_hrnet_softargmax
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision.transforms.v2.functional as VF
from torchvision.io import decode_image

det_model = YOLO("yolov11n-face.pt")

config.defrost()
config.merge_from_file("heatmap_hrnet.yaml")
config.freeze()

model = heatmap_hrnet_softargmax.get_cls_net(config.MODEL.HRNET)
state_dict = load_file('./networks/hrnet_softargmax_roi/ep0200/model.safetensors', device='cuda')
model.load_state_dict(state_dict)
model.cuda()

img = decode_image('Creating an Actor-specific Facial Rig from Performance Capture - 2947688.2947693.pdf.png')[:3]
img = VF.adjust_contrast(img, 2)
img = VF.pad(img, [0, 0, (32-img.shape[-1]%32)%32, (32-img.shape[-2]%32)%32 ]) # pad to stride 32
img = VF.to_dtype(img, dtype=torch.float32, scale=True).cuda()
print(img.shape)
x_lmk = None

yolo_output = det_model.predict(source=img[None], show=False, classes=[0], verbose=False)
img = VF.normalize(img, mean=SynthFaceDataset_RoI.mean, std=SynthFaceDataset_RoI.std)
for bbox in yolo_output[0].boxes.xyxy:
    bbox = torch.round(bbox).to(dtype=torch.int32)

    with torch.no_grad():
        lmk, htmap = model.pred(img, bbox, margin=5, return_heatmap=True)

    print(lmk.shape)
    lmk = lmk.cpu().numpy()
    x = img.cpu().numpy()
    if x_lmk is None:
        print('x:', x.shape)
        x_lmk = SynthFaceDataset_RoI.denorm_and_transpose(x.copy())
    x_lmk = SynthFaceDataset_RoI.draw_landmark_single_img(x_lmk, lmk[0])

    x1, y1, x2, y2 = bbox
    cv2.rectangle(x_lmk, (int(x1), int(y1)), (int(x2), int(y2)), (255, 201, 255), 5)

if x_lmk is None:
    cv2.imwrite('test.jpg', cv2.hconcat([SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy()), SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy())]))
else:
    cv2.imwrite('test.jpg', cv2.hconcat([SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy()), x_lmk]))