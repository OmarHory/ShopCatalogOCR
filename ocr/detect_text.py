from pathlib import Path
import torch
from ocr.models.experimental import attempt_load
from ocr.utils.datasets import LoadImages
from ocr.utils.general import  non_max_suppression, apply_classifier,scale_coords, xyxy2xywh, set_logging
from ocr.utils.torch_utils import select_device, load_classifier, time_synchronized

class TextDetection(object):
    def __init__(self, weights, device_loc='cpu', augment=False ,conf_thres=0.25, iou=0.45, classes=None, agnostic_nms=False):
        self.weights = weights
        self.device_loc = device_loc
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou = iou
        self.classes = classes
        self.agnostic_nms =agnostic_nms 


    def detect(self, source, img_size=640):
        source, weights= source, self.weights
        set_logging()
        device = select_device(self.device_loc)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        dataset = LoadImages(source, img_size=img_size, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    results = []
                    for *xyxy, _, _ in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        results.append(xywh)
        return results
