# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.yolo.utils import LOGGER, ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix
from ultralytics.yolo.utils.downloads import attempt_download


class AutoBackend(nn.Module):

    def __init__(self, weights='yolov8n.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        """
        MultiBackend class for python inference on various platforms using Ultralytics YOLO.

        Args:
            weights (str): The path to the weights file. Default: 'yolov8n.pt'
            device (torch.device): The device to run the model on.
            dnn (bool): Use OpenCV's DNN module for inference if True, defaults to False.
            data (dict): Additional data, optional
            fp16 (bool): If True, use half precision. Default: False
            fuse (bool): Whether to fuse the model or not. Default: True

        Supported formats and their naming conventions:
            | Format                | Suffix           |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | ONNX Runtime          | *.onnx           |
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        
        pt, onnx = str(weights).endswith(".pt"), str(weights).endswith('.onnx')

        fp16 &= pt or onnx or nn_module  # FP16
        nhwc = False  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        model = None  # TODO: resolves ONNX inference, verify effect on other backends
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or nn_module):
            w = attempt_download(w)  # download if not local

        # NOTE: special case: in-memory pytorch model
        if nn_module:
            model = weights.to(device)
            model = model.fuse() if fuse else model
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            stride = max(int(model.stride.max()), 32)  # model stride
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True
        elif pt:  # PyTorch
            from ultralytics.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights if isinstance(weights, list) else w,
                                         device=device,
                                         inplace=True,
                                         fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        else:
            raise NotImplementedError(f"ERROR: '{w}' is not a supported format. For supported formats see "
                                      f"https://docs.ultralytics.com/reference/nn/")

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'yolo/data/datasets/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and the processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt or self.nn_module:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)

        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        """
        Loads the metadata from a yaml file

        Args:
            f: The path to the metadata file.
        """

        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None
