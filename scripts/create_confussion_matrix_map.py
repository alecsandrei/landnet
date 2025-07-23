from __future__ import annotations

import numpy as np
import rasterio


def main(args):
    with rasterio.open(args.input_y) as input_y:
        y = input_y.read(1)
        metadata = input_y.meta.copy()

    with rasterio.open(args.input_yhat) as input_yhat:
        yhat = input_yhat.read(1)

    results = np.zeros(y.shape)
    labels = [
        (yhat == 1) & (y == 1),  # tp = 1
        (yhat == 1) & (y == 0),  # fp = 2
        (yhat == 0) & (y == 0),  # tn = 3
        (yhat == 0) & (y == 1),  # fn = 4
    ]
    for i, label in enumerate(labels, start=1):
        results[label] = i

    with rasterio.open(args.output, 'w', **metadata) as output:
        output.write(results, 1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a confusion matrix map from a raster file.'
    )
    parser.add_argument('input_y', type=str)
    parser.add_argument('input_yhat', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    main(args)
