import os, sys
from configs import config



def validate(image_url):
    if type(image_url) is not str:
        raise ValueError("Image URL must be a string")
    if not os.path.exists(image_url):
        raise ValueError("Image URL doesn't exist.")
    if os.path.splitext(image_url)[-1] not in config["allowed_extensions"]:
        raise ValueError(
            "Extension not allowed, allowed extensions are: {}".format(
                config["allowed_extensions"]
            )
        )

def validate_bulk(image_urls):
    if type(image_urls) is not list:
        raise ValueError("Image URLs must be in a list")
    if type(image_urls[0]) is not str:
        raise ValueError("Image URL must be a string")
    
    image_urls = set(image_urls)

    for image_url in image_urls:
        if not os.path.exists(image_url):
            raise ValueError("Image URL doesn't exist.")

        if os.path.splitext(image_url)[-1] not in config["allowed_extensions"]:
            raise ValueError(
        "Extension not allowed, allowed extensions are: {}".format(
            config["allowed_extensions"]
        )
    )

