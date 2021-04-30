from bs4 import BeautifulSoup
import requests
import os
from datetime import datetime
import re
import sys
from configs import config

today = datetime.now()
date = today.strftime("%d-%m-%Y")
time = today.strftime("%m-%d-%Y__%H:%M:%S__")
URL = "https://www.wowdeals.me/catalogs?page={}&per_page=24"
destination = os.path.join(config["dataset_dir"], date)
if not os.path.exists(destination):
    os.mkdir(destination)
os.chdir(destination)

for i in range(1, 100):
    url = requests.get(URL.format(i))
    if url.status_code != 200:
        print("\nFinished Scraping until page: ", i)
        break

    htmltext = url.text

    soup = BeautifulSoup(htmltext)

    product_names = soup.find_all("div", {"class": "product-name"})
    links_of_catalog = []

    for product in product_names:
        soup_product = BeautifulSoup(str(product))

        link = soup_product.find_all("a", href=True)

        links_of_catalog.append(link[0]["href"])
    if len(links_of_catalog) == 0:
        sys.exit("No links, exiting.")
    print(links_of_catalog)

    new_links = []
    for link in links_of_catalog:
        new_links.append(link + "#Page0")
    del links_of_catalog

    images = []
    for link in new_links:
        url = requests.get(link)
        htmltext = url.text
        soup = BeautifulSoup(htmltext)

        mydivs = soup.find_all("script", {"type": "text/javascript"})
        try:
            string = str(mydivs[3])
        except:
            continue
        regex = r"(https?://[^\s]+)"

        images.append(re.findall(regex, string)[0][:-2])
    del new_links

    for image_link in images:
        LINK = image_link.replace("1.jpg", "{}.jpg")

        for i in range(1, 100):
            response = requests.get(LINK.format(i))
            if response.status_code == 200:
                file_name = time + os.path.basename(LINK.format(i))
                os.system("wget " + LINK.format(i) + " -O " + file_name)
            else:
                break
