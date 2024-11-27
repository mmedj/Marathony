# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import os
from pathlib import Path
import os
import shutil


def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
            ocr = result[1]

    return str(ocr)


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        global label
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p

        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None
                ocr = getOCR(im0, xyxy)
                if ocr.isdigit():
                    label = ocr
                    # Copy image to folder with OCR label as name
                    source_image_path = str(self.data_path)
                    destination_folder = os.path.join(Path("./static/images/Treated"), label)
                    copy_image_to_folder(source_image_path, destination_folder, f'{p.stem}_{frame}.jpg')
        return log_string


def copy_image_to_folder(source_image_path, destination_folder, new_image_name):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_path = os.path.join(destination_folder, new_image_name)
    if os.path.exists(destination_path):
        return False
    shutil.copy(source_image_path, destination_path)
    return True


def delete_images_in_folder(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            file_path = os.path.join(folder_path, file)

            os.remove(file_path)

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = "./model/best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    source = "./static/images/Untreated"
    image_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    for image_file in image_files:
        cfg.source = os.path.join(source, image_file)
        predictor = DetectionPredictor(cfg)
        predictor()
    delete_images_in_folder(source)

reader = easyocr.Reader(['en'])