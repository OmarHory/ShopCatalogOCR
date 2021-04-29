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
            configuration = "-l ara --oem 1 --psm 11"
            text = pytesseract.image_to_string(r, config=configuration)
            results.append(((startX, startY, endX, endY), text))

        orig_image = orig.copy()

        # Moving over the results and display on the image
        for ((start_X, start_Y, end_X, end_Y), text) in results:

            text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
            cv2.rectangle(
                orig_image, (start_X, start_Y), (end_X, end_Y), (0, 0, 255), 2
            )
            cv2.putText(
                orig_image,
                "",
                (start_X, start_Y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        plt.imshow(orig_image)
        plt.savefig("a.jpg", dpi=300)

        return results
