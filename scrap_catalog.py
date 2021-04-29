import os
from datetime import datetime
import argparse


now = datetime.now()  # current date and time
time = now.strftime("%m-%d-%Y__%H:%M:%S__")


parser = argparse.ArgumentParser()
parser.add_argument("--link", help="link.", type=str, required=True)
parser.add_argument(
    "--dest", help="destination directory to write to", type=str, required=True
)
parser.add_argument("--pages_num", help="pages_num", type=int, required=True)

args = parser.parse_args()

LINK = args.link.replace("1.jpg", "{}.jpg")
dataset_dir = "Datasets"

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

destination = os.path.join(dataset_dir, args.dest)

if not os.path.exists(destination):
    os.mkdir(destination)

os.chdir(destination)

for i in range(1, args.pages_num + 1):
    file_name = time + os.path.basename(LINK.format(i))
    os.system("wget " + LINK.format(i) + " -O " + file_name)
