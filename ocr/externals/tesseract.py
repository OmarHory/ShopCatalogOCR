import cv2
import pytesseract
from matplotlib import pyplot as plt


class Tesseract(object):
    def __init__(self):
        pass

    def inference(self, image, boxes):
        orig = image.copy()
        (H, W) = image.shape[:2]

        results = []

        for (centerX, centerY, boxW, boxH) in boxes:
            centerX = centerX * W
            centerY = centerY * H
            boxW = boxW * W
            boxH = boxH * H

            startX = int(centerX - (boxW / 2))
            startY = int(centerY - (boxH / 2))
            endX = int(centerX + (boxW / 2))
            endY = int(centerY + (boxH / 2))

            r = orig[startY:endY, startX:endX]

            configuration_en = "-l eng --oem 1 --psm 11"
            configuration_ar = "-l ara --oem 1 --psm 11"

            text_en = (
                pytesseract.image_to_string(r, config=configuration_en)
                .replace("\n", " ")
                .replace("\f", " ")
            )
            text_ar = (
                pytesseract.image_to_string(r, config=configuration_ar)
                .replace("\n", " ")
                .replace("\f", " ")
            )

            results.append(((startX, startY, endX, endY), text_en, text_ar))

        return results
