
# coding: utf-8
import shlex
import subprocess
import os

import numpy as np

def read_images(imgf, labelf, n):
    with open(imgf, "rb") as f, open(labelf, "rb") as l:
        f.read(16)
        l.read(8)

        images = np.empty((n, 28*28+1))
        for i in range(n):
            # Label
            images[i,0] = ord(l.read(1))
            for x in range(28*28):
                # Image
                images[i,x+1] = ord(f.read(1))
        return images

def load_and_normalize(input_data, input_labels, samples, output_file):
    imgs_norm = read_images(input_data, input_labels, samples)
    # Normalize all columns, except the 'index' column
    imgs_norm[:,1:] = imgs_norm[:,1:] / 255.0
    fmt = ["%u"]  # write the label column as unsigned
    fmt.extend(["%.3f"]*28*28)  # write the image with 3 decimals of precision.
    np.savetxt(output_file, imgs_norm, delimiter=",", fmt=fmt)

def download_and_extract(url, outfile):
    if os.path.isfile(os.path.abspath(outfile)):
        print "Detected", outfile, "Skipping download."
        return
    cmd = "wget -O - %s | gunzip -c > %s" % (url, outfile)
    subprocess.check_output(cmd, shell=True)


if __name__ == "__main__":
    if not os.path.isdir(os.path.abspath('../data/mnist')):
        print "This script is supposed to be execute from within the scripts-dir and assumes that ../data/mnist exists."
        exit(1)
    download_and_extract("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "../data/mnist/train-images-idx3-ubyte")
    download_and_extract("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "../data/mnist/train-labels-idx1-ubyte")
    load_and_normalize("../data/mnist/train-images-idx3-ubyte", "../data/mnist/train-labels-idx1-ubyte", 60000, "../data/mnist/mnist_train_normed.csv")

    download_and_extract("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "../data/mnist/t10k-images-idx3-ubyte")
    download_and_extract("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "../data/mnist/t10k-labels-idx1-ubyte")
    load_and_normalize("../data/mnist/t10k-images-idx3-ubyte", "../data/mnist/t10k-labels-idx1-ubyte", 10000, "../data/mnist/mnist_test_normed.csv")
