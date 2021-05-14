from pathlib import Path
import torch
from ocr.externals.models.experimental import attempt_load
from ocr.externals.utils.datasets import LoadImages
from ocr.externals.utils.general import (
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    set_logging,
)
from ocr.externals.utils.torch_utils import select_device, load_classifier


class TextDetection(object):
    def __init__(
        self,
        weights,
        device_loc="cpu",
        augment=False,
        conf_thres=0.25,
        iou=0.45,
        classes=None,
        agnostic_nms=False,
    ):
        self.weights = weights
        self.device_loc = device_loc
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou = iou
        self.classes = classes
        self.agnostic_nms = agnostic_nms

    def load_model(self):
        weights = self.weights
        set_logging()
        self.device = select_device(self.device_loc)
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        if self.half:
            self.model.half()  # to FP16

    def detect(self, source, img_size=640):
        # if self.model is None:
        #     raise "Model is not loaded."
        try:
            dataset = LoadImages(source, img_size=img_size, stride=self.stride)

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, "module") else self.model.names
            if self.device.type != "cpu":
                self.model(
                    torch.zeros(1, 3, img_size, img_size)
                    .to(self.device)
                    .type_as(next(self.model.parameters()))
                )  # run once
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou,
                    classes=self.classes,
                    agnostic=self.agnostic_nms,
                )

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    s += "%gx%g " % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape
                        ).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        results = []
                        for *xyxy, _, _ in reversed(det):
                            xywh = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )  # normalized xywh
                            results.append(xywh)
            return results
        except:
            raise ValueError("Model Not Loaded.")

