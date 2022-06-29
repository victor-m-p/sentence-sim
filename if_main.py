# imports 
import pandas as pd 
import numpy as np
import argparse 
from os import listdir
from os.path import isfile, join
import datetime

def main(inpath, outpath): 
    pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inpath", required=True, type=str, help="file to process (csv)")
    ap.add_argument("-op", "--outpath", required=True, type=str, help='path to folder for saving output files (txt)')
    args = vars(ap.parse_args())
    main(
        inpath = args['inpath'], 
        outpath = args['outpath'])