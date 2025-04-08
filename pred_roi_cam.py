from safetensors.torch import load_file
from config import config
from config import update_config
import heatmap_hrnet_softargmax
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms.v2.functional as VF
from torchvision.io import decode_image
import matplotlib.pyplot as plt
from dataset import SynthFaceDataset_RoI
import cv2


if __name__ == "__main__":
    det_model = YOLO("yolov11n-face.pt")

    config.defrost()
    config.merge_from_file("heatmap_hrnet.yaml")
    config.freeze()

    model = heatmap_hrnet_softargmax.get_cls_net(config.MODEL.HRNET)
    state_dict = load_file('./networks/hrnet_softargmax_roi/ep0200/model.safetensors', device='cuda')
    model.load_state_dict(state_dict)

    model.cuda()

    print('model load done.')

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while(True):

        ret, frame_orig = cap.read()
        print(frame_orig.shape)
        print(ret)
        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        img = torch.tensor(frame_orig.transpose([2, 0, 1]), dtype=torch.uint8)
        # img = decode_image('thumb_lh.png')
        img = VF.pad(img, [0, 0, (32-img.shape[-1]%32)%32, (32-img.shape[-2]%32)%32 ]) # pad to stride 32
        img = VF.to_dtype(img, dtype=torch.float32, scale=True).cuda()
        print(img.shape)
        x_lmk = None

        yolo_output = det_model.predict(source=img[None], show=False, classes=[0], verbose=False)
        img = VF.normalize(img, mean=SynthFaceDataset_RoI.mean, std=SynthFaceDataset_RoI.std)
        for bbox in yolo_output[0].boxes.xyxy:
            bbox = torch.round(bbox).to(dtype=torch.int32)

            # input = SynthFaceDataset_Heatmap.load_n_normalize('107248.jpg').cuda()
            # lmk, htmap = model.pred(input[None], return_heatmap=True)
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
            cv2.imshow('test', cv2.hconcat([SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy()), SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy())]))
        else:
            
            cv2.imshow('test', cv2.hconcat([SynthFaceDataset_RoI.denorm_and_transpose(img.cpu().numpy()), x_lmk]))
        if cv2.waitKey(1) == ord('q'):
            exit()

        # cv2.imwrite('test_heatmap.jpg', htimg)