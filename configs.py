import os

# TODO: Support environment variables

config = dict(
    text_detection_model="ocr/best.pt",
    img_size=640,
    allowed_extensions=[".jpg", ".png"],
    dataset_dir="/home/omar/Desktop/ShopCatalogOCR/Datasets",
    max_image_size=5 * 1024 * 1024,
    debug=True,
)
