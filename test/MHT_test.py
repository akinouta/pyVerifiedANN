
import numpy as np

from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.guide_search import *
from modules.outsource.MHT import *

vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")[:20]
indexes = range(vectors.shape[0])

hcnng = createHCNNG(vectors, indexes, 5, 20)
gts = get_gts(vectors, hcnng)




