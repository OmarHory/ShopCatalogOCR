from bs4 import BeautifulSoup
import requests
import os
from datetime import datetime
import re
today = datetime.now()
date= today.strftime("%d-%m-%Y")
time = today.strftime("%m-%d-%Y__%H:%M:%S__")


url = requests.get("https://www.wowdeals.me/ae/grand-hypermarket/catalogs/")
htmltext = url.text

soup = BeautifulSoup(htmltext)

product_names = soup.find_all("div", {"class": "product-name"})
links_of_catalog =[]

for product in product_names:
    soup_product = BeautifulSoup(str(product))

    link = soup_product.find_all("a", href=True)

    links_of_catalog.append(link[0]['href'])

print(links_of_catalog)

new_links = []
for link in links_of_catalog:
    new_links.append(link+'#Page0')
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
    regex = r'(https?://[^\s]+)'
    
    images.append(re.findall(regex, string)[0][:-2])
del new_links

dataset_dir = "Datasets"

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

destination = os.path.join(dataset_dir, date)

if not os.path.exists(destination):
    os.mkdir(destination)

os.chdir(destination)

for image_link in images:
    LINK = image_link.replace("1.jpg", "{}.jpg")
    
    for i in range(1, 100):
        response = requests.get(LINK.format(i))
        if response.status_code == 200:
            file_name = time + os.path.basename(LINK.format(i))
            os.system("wget " + LINK.format(i) + " -O " + file_name)
        else:
            break
    
