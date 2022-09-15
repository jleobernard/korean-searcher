import os
import shutil
import logging
import argparse
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import pandas as pd

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='datadir', required=True,
                    help='path to input images')
parser.add_argument('--nb-subs', dest='nb_images', default=100,
                    help='Max learning rate')
parser.add_argument('--max-nb-lines', dest='max_nb_lines', default=3,
                    help='Max number of lines')

args = parser.parse_args()

CHARACTERS_PER_LINE = 10
MAX_HEIGHT = 250
MAX_WIDTH = 500
HEIGHT_VARIATION = 100
WIDTH_VARIATION = 100

template_dir = f"{args.datadir}/templates"
out_dir_path = f"{args.datadir}/out"
fonts_dir_path = f"{args.datadir}/fonts"

logging.info(f"Cleaning output dir {out_dir_path}")
shutil.rmtree(out_dir_path, ignore_errors=True)
os.makedirs(out_dir_path)

logging.info(f"Reading corpus file {args.datadir}/corpus.txt")
with open(f"{args.datadir}/corpus.txt", mode='r', encoding='UTF8') as corpus_file:
    haystack = corpus_file.read()
    len_haystack = len(haystack)

logging.info(f"Listing template files in {template_dir}")
template_files = [f for f in os.listdir(template_dir)]
logging.info(f"{len(template_files)} template files available")

logging.info(f"Listing fonts in {fonts_dir_path}")
fonts = [ImageFont.truetype(f"{fonts_dir_path}/{f}", 60) for f in os.listdir(fonts_dir_path)]

data = []
for i in tqdm(range(int(args.nb_images))):
    nb_lines = random.randint(1, int(args.max_nb_lines))
    template_file = f"{template_dir}/{random.choice(template_files)}"
    font = random.choice(fonts)
    length_text = nb_lines * CHARACTERS_PER_LINE
    start_index = random.randint(0, len_haystack - length_text)
    text = None
    for _ in range(nb_lines):
        start = random.randint(0, len_haystack - CHARACTERS_PER_LINE - 1)
        sub_text = haystack[start: start + random.randint(1, CHARACTERS_PER_LINE)]
        if text:
            text += f"\n{sub_text}"
        else:
            text = sub_text
    template_img = Image.open(template_file)
    w, h = template_img.size
    draw = ImageDraw.Draw(template_img)
    anchor = (random.randint(0, w - MAX_WIDTH), random.randint(0, h - MAX_HEIGHT))
    draw.multiline_text(anchor, text, font=font, align='center', stroke_width=1, spacing=20)
    textbox = draw.textbbox(anchor, text, font=font, align='center', stroke_width=1, spacing=20)
    textbox = (max(0, textbox[0] - random.randint(0, WIDTH_VARIATION)),
               max(0, textbox[1] - random.randint(0, HEIGHT_VARIATION)),
               min(w, textbox[2] + random.randint(0, WIDTH_VARIATION)),
               min(h, textbox[3] + random.randint(0, HEIGHT_VARIATION)))
    cropped = template_img.crop(textbox)
    file_name = f"{i:04}.png"
    cropped.save(f"{out_dir_path}/{file_name}", quality=100)
    data.append([file_name, text])
df = pd.DataFrame(data=data, columns=["file", "text"])
df.to_csv(f"{out_dir_path}/groundtruth.csv", index=False)
