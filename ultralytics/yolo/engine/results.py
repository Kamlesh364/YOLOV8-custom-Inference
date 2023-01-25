import numpy as np
import torch


class Results:
    """
        A class for storing and manipulating inference results.

        Args:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_shape (tuple, optional): Original image size.

        Attributes:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_shape (tuple, optional): Original image size.

        """

    def __init__(self, boxes=None, probs=None, orig_shape=None) -> None:
        self.boxes = Boxes(boxes, orig_shape) if boxes is not None else None  # native size boxes
        # self.masks = Masks(masks, orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = probs.softmax(0) if probs is not None else None
        self.orig_shape = orig_shape
        self.comp = ["boxes", "probs"]

    def __getitem__(self, idx):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item)[idx])
        return r

    def cpu(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).cpu())
        return r

    def numpy(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).numpy())
        return r

    def cuda(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).cuda())
        return r

    def to(self, *args, **kwargs):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).to(*args, **kwargs))
        return r

    def __len__(self):
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            return len(getattr(self, item))

    def __str__(self):
        str_out = ""
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            str_out = str_out + getattr(self, item).__str__()
        return str_out

    def __repr__(self):
        str_out = ""
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            str_out = str_out + getattr(self, item).__repr__()
        return str_out

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. Valid '{name}' object attributes and properties are:

            Attributes:
                boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
                masks (Masks, optional): A Masks object containing the detection masks.
                probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
                orig_shape (tuple, optional): Original image size.
            """)


class Boxes:
    """
    A class for storing and manipulating detection boxes.

    Args:
        boxes (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6). The last two columns should contain confidence and class values.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        boxes (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6).
        orig_shape (numpy.ndarray): Original image size, in the format (height, width).

    Properties:
        xyxy (numpy.ndarray): The boxes in xyxy format.
        conf (numpy.ndarray): The confidence values of the boxes.
        cls (numpy.ndarray): The class values of the boxes.
    """

    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        assert boxes.shape[-1] == 6  # xyxy, conf, cls
        self.boxes = boxes
        self.orig_shape = torch.as_tensor(orig_shape, device=boxes.device) if isinstance(boxes, torch.Tensor) \
            else np.asarray(orig_shape)

    @property
    def xyxy(self):
        return self.boxes[:, :4]

    @property
    def conf(self):
        return self.boxes[:, -2]

    @property
    def cls(self):
        return self.boxes[:, -1]

    def cpu(self):
        boxes = self.boxes.cpu()
        return Boxes(boxes, self.orig_shape)

    def numpy(self):
        boxes = self.boxes.numpy()
        return Boxes(boxes, self.orig_shape)

    def cuda(self):
        boxes = self.boxes.cuda()
        return Boxes(boxes, self.orig_shape)

    def to(self, *args, **kwargs):
        boxes = self.boxes.to(*args, **kwargs)
        return Boxes(boxes, self.orig_shape)

    @property
    def shape(self):
        return self.boxes.shape

    def __len__(self):  # override len(results)
        return len(self.boxes)

    def __str__(self):
        return self.boxes.__str__()

    def __repr__(self):
        return (f"Ultralytics YOLO {self.__class__} masks\n" + f"type: {type(self.boxes)}\n" +
                f"shape: {self.boxes.shape}\n" + f"dtype: {self.boxes.dtype}\n + {self.boxes.__repr__()}")

    def __getitem__(self, idx):
        boxes = self.boxes[idx]
        return Boxes(boxes, self.orig_shape)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. Valid '{name}' object attributes and properties are:

            Attributes:
                boxes (numpy.ndarray): A tensor or numpy array containing the detection boxes,
                    with shape (num_boxes, 6).
                orig_shape (numpy.ndarray): Original image size, in the format (height, width).

            Properties:
                xyxy (numpy.ndarray): The boxes in xyxy format.
                conf (numpy.ndarray): The confidence values of the boxes.
                cls (numpy.ndarray): The class values of the boxes.
            """)


if __name__ == "__main__":
    # test examples
    results = Results(boxes=torch.randn((2, 6)), masks=torch.randn((2, 160, 160)), orig_shape=[640, 640])
    results = results.cuda()
    print("--cuda--pass--")
    results = results.cpu()
    print("--cpu--pass--")
    results = results.to("cuda:0")
    print("--to-cuda--pass--")
    results = results.to("cpu")
    print("--to-cpu--pass--")
    results = results.numpy()
    print("--numpy--pass--")
