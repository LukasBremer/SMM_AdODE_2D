import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('nr', type=str)
args = parser.parse_args()

key = 'NNRec_eta'

if key == 'eta':#shape (#i,#j,#tries,#eta)=(10,10,5,2)
    np.save('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy',np.full((10,10,5,2),np.nan))
elif key == 'NNRec_eta':#shape (#i,#j,#eta,eta/loss)=(10,10,4,2)
    np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eta_sweep'+args.nr+'.npy',np.full((10,10,4,2),np.nan))
elif key == 'NNRec':#shape (#i,#j,#timepoints,#datasets_to_fit)=(10,10,4,2)
    np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps2.npy',np.full((10,10,4,2),np.nan))