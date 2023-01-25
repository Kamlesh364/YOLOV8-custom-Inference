# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """

from pathlib import Path
import cv2

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadPilAndNumpy
from ultralytics.yolo.data.utils import IMG_FORMATS
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_file
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.bs = None
        self.imgsz = None
        self.device = None
        self.classes = self.args.classes
        self.dataset = None
        self.annotator = None
        self.data_path = None
        # self.callbacks = defaultdict(list, {k: v for k, v in callbacks.default_callbacks.items()})  # add callbacks
        # callbacks.add_integration_callbacks(self)

    def preprocess(self, img):
        pass

    def postprocess(self, preds, img, orig_img, classes=None):
        return preds

    def setup_source(self, source=None):
        if not self.model:
            raise Exception("setup model before setting up source!")
        # source
        source, webcam, screenshot, from_img = self.check_source(source)
        if webcam or screenshot:
            raise "Wrong format as source, only Image_type is acceptable."
        # model
        stride, pt = self.model.stride, self.model.pt
        imgsz = check_imgsz(self.args.imgsz, stride=stride, min_dim=2)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if from_img:
            self.dataset = LoadPilAndNumpy(source,
                                           imgsz=imgsz,
                                           stride=stride,
                                           auto=pt,
                                           transforms=getattr(self.model.model, 'transforms', None))
        else:
            self.dataset = LoadImages(source,
                                      imgsz=imgsz,
                                      stride=stride,
                                      auto=pt,
                                      transforms=getattr(self.model.model, 'transforms', None),
                                      vid_stride=self.args.vid_stride)
        self.from_img = True
        self.imgsz = imgsz
        self.bs = bs

    @smart_inference_mode()
    def __call__(self, source=None, model=None, verbose=False, stream=False):
        if stream:
            return self.stream_inference(source, model, verbose)
        else:
            return list(self.stream_inference(source, model, verbose))  # merge list of Result into one

    def predict_cli(self):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(verbose=True)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def stream_inference(self, source=None, model=None, verbose=False):
        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source. Run every time predict is called
        self.setup_source(source)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        for batch in self.dataset:
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=False)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s, self.classes)
            for i in range(len(im)):
                p, im0 = (path[i], im0s[i])
                p = Path(p)

            yield from self.results

    def setup_model(self, model):
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
        self.device = device
        self.model.eval()
        
    def check_source(self, source):
        source = source if source is not None else self.args.source
        webcam, screenshot, from_img = False, False, False
        if isinstance(source, (str, int, Path)):  # int for local usb carame
            source = str(source)
            is_file = Path(source).suffix[1:] in (IMG_FORMATS)
            webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
            screenshot = source.lower().startswith('screen')
            if is_file:
                source = check_file(source)  # download
        else:
            from_img = True
        return source, webcam, screenshot, from_img
