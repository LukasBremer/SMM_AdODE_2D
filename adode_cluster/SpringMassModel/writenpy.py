import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('nr', type=str)
args = parser.parse_args()

np.save('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy',np.full((10,10,5,2),np.nan))