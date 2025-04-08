import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_iou
import torchvision.transforms.v2.functional as VF
from torchvision.transforms.v2.functional._geometry import _get_inverse_affine_matrix
from torchvision.transforms.v2.functional._geometry import _center_crop_compute_crop_anchor
from torchvision.io import decode_image
import torchvision.transforms.v2 as T
from torchvision.io import read_image
import numpy as np
import json
import os
import re
import cv2

# from ms synthmocap github.
face_conn = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20],
    [20, 21],
    [22, 23],
    [23, 24],
    [24, 25],
    [25, 26],
    [27, 28],
    [28, 29],
    [29, 30],
    [31, 32],
    [32, 33],
    [33, 34],
    [34, 35],
    [36, 37],
    [37, 38],
    [38, 39],
    [39, 40],
    [40, 41],
    [41, 36],
    [42, 43],
    [43, 44],
    [44, 45],
    [45, 46],
    [46, 47],
    [47, 42],
    [48, 49],
    [49, 50],
    [50, 51],
    [51, 52],
    [52, 53],
    [53, 54],
    [54, 55],
    [55, 56],
    [56, 57],
    [57, 58],
    [58, 59],
    [59, 48],
    [60, 61],
    [61, 62],
    [62, 63],
    [63, 64],
    [64, 65],
    [65, 66],
    [66, 67],
    [67, 60],
]

face_right_lmks = [
    0, 1, 2, 3, 4, 5, 6, 7, # jaw
    17, 18, 19, 20, 21, # eyebrow
    36, 37, 38, 39, 40, 41, 68, # eye
    31, 32, # nose
    48, 49, 50, # mouth outer
    59, 58,
    60, 61, 67, # mouse inner
]

face_left_lmks = [ # corres to right(mirrored)
    16, 15, 14, 13, 12, 11, 10, 9, # jaw
    26, 25, 24, 23, 22, # eyebrow
    45, 44, 43, 42, 47, 46, 69, # eye
    35, 34, # nose
    54, 53, 52, # mouth outer
    55, 56,
    64, 63, 65, # mouse inner
]

def build_bbox():
    from ultralytics import YOLO
    
    det_model = YOLO("yolov11n-face.pt")
    img_dir = './synth_face/'
    img_re = re.compile(r"img_([0-9]*_[0-9]*)\.jpg")

    # img = decode_image('107248.jpg')[None]
    # img = VF.pad(img, [0, 0, (32-img.shape[-1]%32)%32, (32-img.shape[-2]%32)%32 ])
    # img = VF.to_dtype(img, dtype=torch.float32, scale=True)
    # yolo_output = det_model.predict(source=img, show=False, classes=[0], verbose=False)
    # yolo_out_xyxy = yolo_output[0].boxes.xyxy
    # tv_bb = tv_tensors.BoundingBoxes(yolo_out_xyxy, format='xyxy', canvas_size=(img.shape[-2], img.shape[-1]))
    # img = VF.affine(img       , 35, [0, 0], 1.1, [15, 15])
    # tv_bb = VF.affine(tv_bb       , 35, [0, 0], 1.1, [15, 15])
    # img = VF.to_dtype(img, dtype=torch.uint8, scale=True).cpu().numpy()[0]
    # img = img.transpose([1, 2, 0]).copy()
    # print(img.shape)
    # print(tv_bb)
    # for i in range(len(yolo_out_xyxy)):
    #     x1, y1, x2, y2 = tv_bb[i]
    #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)
    # cv2.imwrite('yoloimg.jpg', img)
    # print(yolo_out_xyxy)

    for f in os.listdir(os.path.join(img_dir)):
        m: re.Match = img_re.match(f)
        if m:
            fidx = m.group(1)
            img = decode_image(os.path.join(img_dir,f))
            img = VF.pad(img, [0, 0, (32-img.shape[-1]%32)%32, (32-img.shape[-2]%32)%32 ]) # pad to stride 32
            img = VF.to_dtype(img, dtype=torch.float32, scale=True)

            yolo_output = det_model.predict(source=img[None], show=False, classes=[0], verbose=False)
            yolo_out_xyxy = yolo_output[0].boxes.xyxy

            with open(os.path.join(img_dir,"metadata_"+fidx+".json"), 'r') as ff:
                meta = json.loads(ff.read())
            landmarks = torch.Tensor(meta["landmarks"]["2D"])
            min_xy = landmarks.min(dim=0).values
            max_xy = landmarks.max(dim=0).values
            lmk_xyxy = tv_tensors.BoundingBoxes(torch.cat([min_xy, max_xy])[None], format='xyxy', canvas_size=(img.shape[-2], img.shape[-1]), device=yolo_out_xyxy.device)

            # test code
            # img = VF.to_dtype(img, dtype=torch.uint8, scale=True).cpu().numpy()
            # img = img.transpose([1, 2, 0]).copy()
            # for i in range(len(yolo_out_xyxy)):
            #     x1, y1, x2, y2 = yolo_out_xyxy[i]
            #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)
            # for i in range(len(lmk_xyxy)):
            #     x1, y1, x2, y2 = lmk_xyxy[i]
            #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 201, 255), 5)
            # cv2.imwrite(f'test_{f}.jpg', img)
            # exit()

            # if len(yolo_out_xyxy) > 1:
            #     print(f)
            #     img = VF.to_dtype(img, dtype=torch.uint8, scale=True).cpu().numpy()
            #     img = img.transpose([1, 2, 0]).copy()
            #     for i in range(len(yolo_out_xyxy)):
            #         x1, y1, x2, y2 = yolo_out_xyxy[i]
            #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201+i*20, 219), 5)
            #     for i in range(len(lmk_xyxy)):
            #         x1, y1, x2, y2 = lmk_xyxy[i]
            #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 201, 255), 5)
            #     cv2.imwrite(f'multi_{f}.jpg', img)
            #     exit()

            if len(yolo_out_xyxy) == 0:
                print(f)
                # make pseudo yolo output
                yolo_out_xyxy = lmk_xyxy[0].clone()
            else:
                iou = box_iou(yolo_out_xyxy, lmk_xyxy) # n, 1
                if len(yolo_out_xyxy) > 1:
                    print(f)
                    print(iou)
                yolo_out_xyxy = yolo_out_xyxy[iou[:, 0].max(dim=0).indices] # 1
            
            with open(os.path.join(img_dir,"bbox_"+fidx+".json"), 'w') as ff:
                d = {}
                d['bbox'] = yolo_out_xyxy.cpu().numpy().tolist()
                ff.write(json.dumps(d))






def draw_landmark_idxs(
    img: np.ndarray,
    ldmks_2d: np.ndarray,
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    img_size = (img.shape[1], img.shape[0])
    for i, ldmk in enumerate(ldmks_2d.astype(int)):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.putText(img, str(i), tuple(ldmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness)


def draw_landmarks(
    img: np.ndarray,
    ldmks_2d: np.ndarray,
    connectivity: list[list[int]],
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Drawing dots on an image."""
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if np.any(np.isnan(ldmks_2d)):
        raise ValueError("NaNs in landmarks")

    img_size = (img.shape[1], img.shape[0])

    ldmk_connection_pairs = ldmks_2d[np.asarray(connectivity).astype(int)].astype(int)
    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
    for i, (p_0, p_1) in enumerate(ldmk_connection_pairs):
        cv2.line(
            img,
            tuple(p_0),
            tuple(p_1),
            (int(color[0]), int(color[1]), int(color[2])),
            thickness,
            cv2.LINE_AA,
        )

    for ldmk in ldmks_2d.astype(int):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.circle(img, tuple(ldmk + 1), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(
                img,
                tuple(ldmk),
                thickness + 1,
                (int(color[0]), int(color[1]), int(color[2])),
                -1,
                cv2.LINE_AA,
            )

class SynthFaceDataset(Dataset):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_re = re.compile(r"img_([0-9]*_[0-9]*)\.jpg")
    num_lmk = 70



    def __init__(self, img_dir, augmentation=True):
        self.img_dir = img_dir
        self.normalize = T.Normalize(mean=self.mean, std=self.std)
        self.color_jitter = T.RandomPhotometricDistort()
        self.posterize = T.RandomPosterize(3)
        self.blur = T.GaussianBlur(13, (0.1, 5))
        self.noise = T.GaussianNoise(sigma=0.05)
        self.sharp = T.RandomAdjustSharpness(2)
        self.gray = T.RandomGrayscale(0.2)
        self.jpeg = T.JPEG([1, 50])

        self.augmentation = augmentation
        self.imgs = []
        self.metas = []
        for f in os.listdir(os.path.join(img_dir)):
            m: re.Match = self.img_re.match(f)
            if m:
                fidx = m.group(1)
                self.imgs.append(os.path.join(img_dir,f))
                self.metas.append(os.path.join(img_dir,"metadata_"+fidx+".json"))

    @classmethod
    def denorm_and_transpose(cls, x):
        assert len(x.shape) == 3
        x = x.transpose(1, 2, 0) # 3, h, w => h, w, 3
        x = x * np.array(cls.std)
        x = x + np.array(cls.mean)
        x = (x * 255).astype(np.uint8)
        return x

    @classmethod
    def draw_landmark(cls, x: np.ndarray, lmk: np.ndarray):
        x = cls.denorm_and_transpose(x)
        x_lmk = x.copy()
        draw_landmarks(x_lmk, lmk, face_conn)

        return x, x_lmk

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        meta_path = self.metas[idx]

        image = read_image(img_path)[:3].type(torch.float32) / 255.0
        with open(meta_path, 'r') as f:
            meta = json.loads(f.read())
        landmarks = torch.Tensor(meta["landmarks"]["2D"])

        if self.augmentation:
            rot_deg = np.random.uniform(-60, 60)
            translate = np.random.uniform(-120, 120, size=2).astype(int).tolist()
            scale = np.random.uniform(0.5, 1.3)
            shear = np.random.uniform(-35, 35, size=2).tolist()

            # flip = np.random.uniform() < 0.5
            image        = VF.affine(image       , rot_deg, translate, scale, shear)
            affine_vector = _get_inverse_affine_matrix([image.shape[-1]/2, image.shape[-2]/2], rot_deg, translate, scale, shear, inverted=False)
            affine = (
                torch.tensor(
                    affine_vector,
                    dtype=image.dtype,
                    device=image.device,
                )
                .reshape(2, 3)
                .T
            )
            landmarks = torch.concat([landmarks, torch.ones_like(landmarks[:, :1])], dim=-1)
            landmarks =  landmarks@affine
            landmarks = landmarks[:, :2]
            
            top, left = _center_crop_compute_crop_anchor(448, 448, image.shape[-2], image.shape[-1])
            image        = VF.center_crop(image       , [448, 448])
            landmarks -= torch.Tensor([left, top])

            image = VF.to_dtype(image, torch.float32, True)
            image = self.noise(image)
            image = VF.to_dtype(image, torch.uint8, True)
            image = self.color_jitter(image)
            image = self.posterize(image)
            if np.random.uniform() < 0.5:
                image = self.blur(image)
            if np.random.uniform() < 0.5:
                image = self.jpeg(image)
            image = self.sharp(image)
            image = self.gray(image)
            if np.random.uniform() < 0.5:
                image = VF.horizontal_flip(image)
                landmarks[:, 0] = image.shape[-1] - landmarks[:, 0]
                landmarks_temp = landmarks.clone()
                landmarks_temp[face_right_lmks] = landmarks[face_left_lmks]
                landmarks_temp[face_left_lmks] = landmarks[face_right_lmks]
                landmarks = landmarks_temp
            # if flip:
            #     image        = VF.hflip(image       )
        

        # image = self.transform(image)
        landmarks /= 224.0 # normalize to 0 ~ 2
        landmarks -= 1.0  # normalize to -1 ~ 1
        image = VF.to_dtype(image, torch.float32, True)
        image = self.normalize(image)

        return image, landmarks # (3, h, w), (70, 2)



class SynthFaceDataset_RoI(Dataset):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_re = re.compile(r"img_([0-9]*_[0-9]*)\.jpg")
    num_lmk = 70
    img_size = 256

    def __init__(self, img_dir, augmentation=True):
        self.img_dir = img_dir
        self.normalize = T.Normalize(mean=self.mean, std=self.std)
        self.color_jitter = T.RandomPhotometricDistort()
        self.posterize = T.RandomPosterize(3)
        self.blur = T.GaussianBlur(13, (0.1, 5))
        self.noise = T.GaussianNoise(sigma=0.05)
        self.sharp = T.RandomAdjustSharpness(2)
        self.gray = T.RandomGrayscale(0.2)
        self.jpeg = T.JPEG([1, 50])
        # self.yolo = YOLO("yolov11n-face.pt")

        self.augmentation = augmentation
        self.imgs = []
        self.metas = []
        self.bboxs = []
        for f in os.listdir(os.path.join(img_dir)):
            m: re.Match = self.img_re.match(f)
            if m:
                fidx = m.group(1)
                self.imgs.append(os.path.join(img_dir,f))
                self.metas.append(os.path.join(img_dir,"metadata_"+fidx+".json"))
                self.bboxs.append(os.path.join(img_dir,"bbox_"+fidx+".json"))

    @classmethod
    def denorm_and_transpose(cls, x):
        assert len(x.shape) == 3
        x = x.transpose(1, 2, 0) # 3, h, w => h, w, 3
        x = x * np.array(cls.std)
        x = x + np.array(cls.mean)
        x = (x * 255).astype(np.uint8)
        return x
    
    @classmethod
    def resize_img(cls, x: torch.Tensor):
        scale = cls.img_size / max(x.shape[-2:])

        img = VF.resize(x, (round(x.shape[-2]*scale), round(x.shape[-1]*scale)), VF.InterpolationMode.BILINEAR)
        h, w = img.shape[-2:]
        pad = cls.img_size - min(img.shape[-2:])
        hpad = pad // 2
        if h > w: # need to pad w
            img = VF.pad(img, [hpad, 0, pad-hpad, 0]) # l t r b
            padded = torch.Tensor([hpad, 0])
        else: # need to pad h
            img = VF.pad(img, [0, hpad, 0, pad-hpad]) # l t r b
            padded = torch.Tensor([0, hpad])
        
        return img, scale, padded

    @classmethod
    def draw_landmark_single_img(cls, x: np.ndarray, lmk: np.ndarray):
        x_lmk = x.copy()
        draw_landmarks(x_lmk, lmk, face_conn)

        return x_lmk

    @classmethod
    def draw_landmark(cls, x: np.ndarray, lmk: np.ndarray):
        x = cls.denorm_and_transpose(x)
        x_lmk = x.copy()
        draw_landmarks(x_lmk, lmk, face_conn)

        return x, x_lmk


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        meta_path = self.metas[idx]
        bbox_path = self.bboxs[idx]

        image = decode_image(img_path)[:3].type(torch.float32) / 255.0
        with open(meta_path, 'r') as f:
            meta = json.loads(f.read())
        landmarks = torch.Tensor(meta["landmarks"]["2D"])

        with open(bbox_path, 'r') as f:
            bbox_meta = json.loads(f.read())
        bbox = tv_tensors.BoundingBoxes(bbox_meta["bbox"], format='xyxy', canvas_size=(image.shape[-2], image.shape[-1]))

        if self.augmentation:
            rot_deg = np.random.uniform(-75, 75)
            translate = np.random.uniform(-5, 5, size=2).astype(int).tolist()
            scale = np.random.uniform(0.5, 1.2)
            shear = np.random.uniform(-20, 20, size=2).tolist()

            # flip = np.random.uniform() < 0.5
            image        = VF.affine(image       , rot_deg, translate, scale, shear)
            bbox        = VF.affine(bbox       , rot_deg, translate, scale, shear)

            # yolo versoin bbox
            # with torch.no_grad():
            #     yolo_output = self.yolo.predict(source=image[None], show=False, classes=[0], verbose=False, device='cpu')
            #     bbox = yolo_output[0].boxes.xyxy.cpu()

            affine_vector = _get_inverse_affine_matrix([image.shape[-1]/2, image.shape[-2]/2], rot_deg, translate, scale, shear, inverted=False)
            affine = (
                torch.tensor(
                    affine_vector,
                    dtype=image.dtype,
                    device=image.device,
                )
                .reshape(2, 3)
                .T
            )
            landmarks = torch.concat([landmarks, torch.ones_like(landmarks[:, :1])], dim=-1)
            landmarks =  landmarks@affine
            landmarks = landmarks[:, :2]

            min_xy = landmarks.min(dim=0).values
            max_xy = landmarks.max(dim=0).values
            lmk_xyxy = torch.cat([min_xy, max_xy])

            if np.random.uniform() < 0.5:
                bbox = lmk_xyxy
                xy_random = torch.rand(size=(2,))*30 - 15 # negative make bbox larger
                xy_random_2 = torch.rand(size=(2,))*30 - 15 # positive make bbox larger
                random_xyxy  = torch.cat([xy_random, xy_random_2])
                bbox = torch.clip(bbox+random_xyxy, min=0)
            else:
                bbox = bbox[0]

            # if len(bbox) == 0:
            #     bbox = lmk_xyxy
            # else:
            #     iou = box_iou(bbox, lmk_xyxy[None])
            #     maxval = iou[:, 0].max(dim=0)
            #     if maxval.values < 0.5:
            #         bbox = lmk_xyxy
            #     else:
            #         bbox = bbox[maxval.indices]
            

            bbox = torch.round(torch.Tensor(bbox)).type(dtype=torch.int32)
            image        = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            landmarks -= torch.Tensor(bbox[:2])
            image, lmk_scale, padded = self.resize_img(image)
            landmarks *= lmk_scale
            landmarks += padded


            image = VF.to_dtype(image, torch.float32, True)
            image = self.noise(image)
            image = VF.to_dtype(image, torch.uint8, True)
            image = self.color_jitter(image)
            image = self.posterize(image)
            if np.random.uniform() < 0.5:
                image = self.blur(image)
            if np.random.uniform() < 0.5:
                image = self.jpeg(image)
            image = self.sharp(image)
            image = self.gray(image)
            if np.random.uniform() < 0.5:
                image = VF.horizontal_flip(image)
                landmarks[:, 0] = image.shape[-1] - landmarks[:, 0]
                landmarks_temp = landmarks.clone()
                landmarks_temp[face_right_lmks] = landmarks[face_left_lmks]
                landmarks_temp[face_left_lmks] = landmarks[face_right_lmks]
                landmarks = landmarks_temp
            # if flip:
            #     image        = VF.hflip(image       )
        

        # image = self.transform(image)
        # landmarks /= 224.0 # normalize to 0 ~ 2
        # landmarks -= 1.0  # normalize to -1 ~ 1
        image = VF.to_dtype(image, torch.float32, True)
        image = self.normalize(image)

        return image, landmarks # (3, h, w), (70, 2)



class SynthFaceDataset_Heatmap(Dataset):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_re = re.compile(r"img_([0-9]*_[0-9]*)\.jpg")
    num_lmk = 70


    def __init__(self, img_dir, augmentation=True):
        self.img_dir = img_dir
        self.normalize = T.Normalize(mean=self.mean, std=self.std)
        self.color_jitter = T.RandomPhotometricDistort()
        self.posterize = T.RandomPosterize(3)
        self.blur = T.GaussianBlur(13, (0.1, 5))
        self.noise = T.GaussianNoise(sigma=0.05)
        self.sharp = T.RandomAdjustSharpness(2)
        self.gray = T.RandomGrayscale(0.2)
        self.jpeg = T.JPEG([1, 50])

        self.augmentation = augmentation
        self.imgs = []
        self.metas = []
        for f in os.listdir(os.path.join(img_dir)):
            m: re.Match = self.img_re.match(f)
            if m:
                fidx = m.group(1)
                self.imgs.append(os.path.join(img_dir,f))
                self.metas.append(os.path.join(img_dir,"metadata_"+fidx+".json"))

    @classmethod
    def denorm_and_transpose(cls, x):
        assert len(x.shape) == 3
        x = x.transpose(1, 2, 0) # 3, h, w => h, w, 3
        x = x * np.array(cls.std)
        x = x + np.array(cls.mean)
        x = (x * 255).astype(np.uint8)
        return x

    @classmethod
    def draw_landmark(cls, x: np.ndarray, lmk: np.ndarray):
        x = cls.denorm_and_transpose(x)
        x_lmk = x.copy()
        draw_landmarks(x_lmk, lmk, face_conn)

        return x, x_lmk

    def make_heatmap(self, heatmap_size, img_size, lmk):
        # from HRformer pose/mmpose/datasets/pipelines/top_down_transform.py
        sigma = 1.5
        W, H = heatmap_size

        target_weight = np.ones((len(lmk), 1), dtype=np.float32)
        target = np.zeros((len(lmk), H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3
        for joint_id in range(len(lmk)):
            feat_stride = img_size / [W, H]
            mu_x = int(lmk[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(lmk[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] > 0.5:
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, None]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], W)
                img_y = max(0, ul[1]), min(br[1], H)

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target

    def __len__(self):
        return len(self.imgs)
    
    @classmethod
    def load_n_normalize(cls, path, target_size=448):
        image = read_image(path)[:3].type(torch.float32) / 255.0
        image = VF.resize(image, [target_size, target_size])
        return VF.normalize(image, mean=cls.mean, std=cls.std)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        meta_path = self.metas[idx]

        image = decode_image(img_path)[:3].type(torch.float32) / 255.0
        with open(meta_path, 'r') as f:
            meta = json.loads(f.read())
        landmarks = torch.Tensor(meta["landmarks"]["2D"])

        if self.augmentation:
            rot_deg = np.random.uniform(-60, 60)
            translate = np.random.uniform(-120, 120, size=2).astype(int).tolist()
            scale = np.random.uniform(0.5, 1.5)
            shear = np.random.uniform(-35, 35, size=2).tolist()
            # shear = [0, 0]

            # flip = np.random.uniform() < 0.5
            image        = VF.affine(image       , rot_deg, translate, scale, shear)
            affine_vector = _get_inverse_affine_matrix([image.shape[-1]/2, image.shape[-2]/2], rot_deg, translate, scale, shear, inverted=False)
            affine = (
                torch.tensor(
                    affine_vector,
                    dtype=image.dtype,
                    device=image.device,
                )
                .reshape(2, 3)
                .T
            )
            landmarks = torch.concat([landmarks, torch.ones_like(landmarks[:, :1])], dim=-1)
            landmarks =  landmarks@affine
            landmarks = landmarks[:, :2] 
            
            top, left = _center_crop_compute_crop_anchor(448, 448, image.shape[-2], image.shape[-1])
            image        = VF.center_crop(image       , [448, 448])
            landmarks -= torch.Tensor([left, top])

            image = VF.to_dtype(image, torch.float32, True)
            image = self.noise(image)
            image = VF.to_dtype(image, torch.uint8, True)
            image = self.color_jitter(image)
            image = self.posterize(image)
            if np.random.uniform() < 0.5:
                image = self.blur(image)
            if np.random.uniform() < 0.5:
                image = self.jpeg(image)
            image = self.sharp(image)
            image = self.gray(image)
            if np.random.uniform() < 0.5:
                image = VF.horizontal_flip(image)
                landmarks[:, 0] = image.shape[-1] - landmarks[:, 0]
                landmarks_temp = landmarks.clone()
                landmarks_temp[face_right_lmks] = landmarks[face_left_lmks]
                landmarks_temp[face_left_lmks] = landmarks[face_right_lmks]
                landmarks = landmarks_temp

        
        image = VF.to_dtype(image, torch.float32, True)
        image = self.normalize(image)

        # image = self.transform(image)
        # do not normalize(we use heatmap)
        # landmarks /= 224.0 # normalize to 0 ~ 2
        # landmarks -= 1.0  # normalize to -1 ~ 1

        heatmap = self.make_heatmap((image.shape[-1]//4, image.shape[-2]//4), np.array([image.shape[-1], image.shape[-2]]), landmarks)

        return image, heatmap, landmarks # (3, h, w), (3, h//4, w//4), (70, 2)

if __name__ == "__main__":
    # build_bbox()
    # exit()

    dset = SynthFaceDataset("./synth_face")
    img, lmk = dset[3]
    lmk = lmk.reshape(-1, 2)
    lmk += 1
    lmk *= 224

    print(img.shape)
    print(lmk.shape)

    
    img = (img.numpy() * np.array(dset.std)[:, None, None] + np.array(dset.mean)[:, None, None]) * 255.0
    img = img.astype(np.uint8)
    img = img.transpose(1, 2, 0)
    landmark_img = img.copy()
    print(img.shape)
    draw_landmarks(landmark_img, lmk.numpy(), face_conn)

    cv2.imwrite('test_lmk.png', cv2.hconcat([img, landmark_img]))
    # cv2.waitKey(0)

    dset = SynthFaceDataset_Heatmap("./synth_face")
    img, htmap, lmk = dset[3]
    print('---- heatmap ----')
    print(img.shape)
    print(htmap.shape)
    print(lmk.shape)
    print(htmap.max(), htmap.min())

    lmk = lmk.reshape(-1, 2)
    
    img = (img.numpy() * np.array(dset.std)[:, None, None] + np.array(dset.mean)[:, None, None]) * 255.0
    img = img.astype(np.uint8)
    img = img.transpose(1, 2, 0)
    htmap = htmap * 255
    htmap = htmap.astype(np.uint8)
    htmap_img = cv2.hconcat(htmap[[0, 1, 2, 3, 4]])
    htmap_img = cv2.vconcat([htmap_img, cv2.hconcat(htmap[[5, 6, 7, 8, 9]])])

    landmark_img = img.copy()
    print(img.shape)
    draw_landmarks(landmark_img, lmk.numpy(), face_conn)
    draw_landmark_idxs(landmark_img, lmk.numpy(), color=(0, 0, 255))
    cv2.imwrite('test_htmap.png', htmap_img)
    cv2.imwrite('test_htmap_lmk.png', cv2.hconcat([img, landmark_img]))
    # cv2.waitKey(0)

    dset = SynthFaceDataset_RoI("./synth_face")
    img, lmk = dset[3]

    print(img.shape)
    print(lmk.shape)
    print(lmk)

    
    img = (img.numpy() * np.array(dset.std)[:, None, None] + np.array(dset.mean)[:, None, None]) * 255.0
    img = img.astype(np.uint8)
    img = img.transpose(1, 2, 0)
    landmark_img = img.copy()
    print(img.shape)
    draw_landmarks(landmark_img, lmk.numpy(), face_conn)

    cv2.imwrite('test_lmk_roi.png', cv2.hconcat([img, landmark_img]))