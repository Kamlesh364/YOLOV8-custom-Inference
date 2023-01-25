# Ultralytics YOLO 🚀, GPL-3.0 license

from glob import glob
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import ROOT

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes

class LoadImages:
    # YOLOv8 image dataloader, i.e. `python detect.py --source image.jpg`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob(p, recursive=True)))  # glob
            elif osp.isdir(p):
                files.extend(sorted(glob(osp.join(p, '*.*'))))  # dir
            elif osp.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images # + videos
        self.nf = ni  # number of files
        # self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        # if any(videos):
        #     self._new_video(videos[0])  # new video
        # else:
        #     self.cap = None
        self.cap = None
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        im0 = cv2.imread(path)  # BGR
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640, stride=32, auto=True, transforms=None):
        if not isinstance(im0, list):
            im0 = [im0]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.mode = 'image'
        # generate fake paths
        self.paths = [f"image{i}.jpg" for i in range(len(self.im0))]

    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def _single_preprocess(self, im, auto):
        if self.transforms:
            im = self.transforms(im)  # transforms
        else:
            im = LetterBox(self.imgsz, auto=auto, stride=self.stride)(image=im)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        auto = all(x.shape == self.im0[0].shape for x in self.im0) and self.auto
        im = [self._single_preprocess(im, auto) for im in self.im0]
        im = np.stack(im, 0) if len(im) > 1 else im[0][None]
        self.count += 1
        return self.paths, im, self.im0, None, ''

    def __iter__(self):
        self.count = 0
        return self


if __name__ == "__main__":
    img = cv2.imread(str(ROOT / "assets/bus.jpg"))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
