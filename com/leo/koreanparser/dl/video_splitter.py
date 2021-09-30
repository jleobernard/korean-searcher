import argparse
import os

import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', dest='in_file',
                    help='path to input video file')
parser.add_argument('-o', '--out-dir', dest='out_dir',
                    help='path to output directory')
parser.add_argument('-p', '--prefix', dest='prefix',
                    help='path to output directory')
args = parser.parse_args()
# Opens the Video file
out_dir = args.out_dir
ensure_dir(out_dir)
cap = cv2.VideoCapture(args.in_file)
NB_FRAMES_SKIPPED = 30
i=0
while(cap.isOpened()):
    for j in range(0, NB_FRAMES_SKIPPED):
        ret, frame = cap.read()
        if ret == False:
            exit(0)
    cv2.imwrite(f"{out_dir}/{args.prefix}-{str(i)}.jpg", frame)
    i += 1
    if i % 10 == 0:
        print(f"Exported {i} frames")
cap.release()
cv2.destroyAllWindows()
