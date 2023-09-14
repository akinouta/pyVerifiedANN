import numpy as np

from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs

vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")
indexes = range(20)
for edge in createHCNNG(vectors,indexes,5,20):
    print(edge)