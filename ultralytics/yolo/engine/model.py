# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (DetectionModel, attempt_load_one_weight)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, predictor classes
MODEL_MAP = {
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionPredictor'}


class YOLO:
    """
    YOLO

    A python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, model='yolov8n.yaml', type="v8") -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        self.type = type
        self.ModelClass = None  # model class
        self.PredictorClass = None  # predictor class
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

        # Load or create new YOLO model
        load_methods = {'.pt': self._load, '.yaml': self._new}
        suffix = Path(model).suffix
        if suffix in load_methods:
            {'.pt': self._load, '.yaml': self._new}[suffix](model)
        else:
            raise NotImplementedError(f"'{suffix}' model loading not implemented")

    def __call__(self, source=None, stream=False, verbose=False, **kwargs):
        return self.predict(source, stream, verbose, **kwargs)

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
        self.task = "detect"
        self.ModelClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize
        self.cfg = cfg

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.ckpt_path = weights
        self.task = "detect"
        self.overrides = self.model.args
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)

    def reset(self):
        """
        Resets the model modules.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    def fuse(self):
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, verbose=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            verbose (bool): Whether to print verbose information or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (dict): The prediction results.
        """
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        overrides["save"] = kwargs.get("save", False)  # not save files by default
        if not self.predictor:
            self.predictor = self.PredictorClass(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source=source, stream=stream, verbose=verbose)

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self.model.to(device)

    def _assign_ops_from_task(self, task):
        model_class, pred_lit = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))

        return model_class, predictor_class

    @property
    def names(self):
        """
         Returns class names of the loaded model.
        """
        return self.model.names

    @staticmethod
    def _reset_ckpt_args(args):
        args.pop("project", None)
        args.pop("name", None)
        args.pop("exist_ok", None)
        args.pop("resume", None)
        args.pop("batch", None)
        args.pop("epochs", None)
        args.pop("cache", None)
        args.pop("save_json", None)
        args.pop("half", None)
        args.pop("v5loader", None)

        # set device to '' to prevent from auto DDP usage
        args["device"] = ''
