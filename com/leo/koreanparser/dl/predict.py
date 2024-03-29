import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from com.leo.koreanparser.dl.conf import TARGET_HEIGHT, TARGET_WIDTH
from com.leo.koreanparser.dl.model import get_model, get_bb_from_bouding_boxes
from com.leo.koreanparser.dl.utils.data_utils import read_image, SubsDataset, show_corner_bb
from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device
from com.leo.koreanparser.dl.utils.train_utils import do_load_model, do_lod_specific_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', dest='weights_path',
                        help='path to model weights', required=True)
    parser.add_argument('-f', '--file', dest='file',
                        help='path to file to analyze', required=True)
    parser.add_argument('-t', '--threshold', dest='threshold',
                        help='score threshold to consider the image as containing subs', default="0.5")
    args = vars(parser.parse_args())
    weights_path = args['weights_path']
    threshold = float(args["threshold"])

    # resizing test image
    im = read_image(args['file'])
    size = (TARGET_WIDTH, TARGET_HEIGHT)
    im = cv2.resize(im, size)
    cv2.imwrite('/tmp/tmp.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    test_ds = SubsDataset(pd.DataFrame([{'path': '/tmp/tmp.jpg'}])['path'], pd.DataFrame([{'bb': np.array([0, 0, 0, 0])}])['bb'],pd.DataFrame([{'y': [0]}])['y'])
    x, y_class, y_bb = test_ds[0]
    xx = to_best_device(torch.FloatTensor(x[None,]))

    model = get_model(eval=True)
    do_lod_specific_model(weights_path, model)

    out_class, out_bb = model(xx)

    bb_hat = out_bb.detach().cpu()
    bounding_boxes = get_bb_from_bouding_boxes(bb_hat, height=TARGET_HEIGHT, width=TARGET_WIDTH)
    class_hat = torch.sigmoid(out_class.detach().cpu()).numpy()
    if class_hat[0][0] >= threshold:
        print(f"L'image contient des sous-titres ({class_hat[0][0]})")
    else:
        print(f"L'image ne contient pas de sous-titres ({class_hat[0][0]})")
    print(f"Predicted bounding box is {bounding_boxes[0]}")
    show_corner_bb(im, bounding_boxes[0])
