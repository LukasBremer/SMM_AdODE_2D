import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('nr', type=str)
args = parser.parse_args()

key = 'NNRec_eta'

# if key == 'eta':#shape (#i,#j,#tries,#eta)=(10,10,5,2)
#     np.save('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy',np.full((10,10,5,2),np.nan))
# elif key == 'NNRec_eta':#shape (#i,#j,#eta,eta/loss)=(10,10,4,2)
#     np.save('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy',np.full((10,10,4,2),np.nan))
# elif key == 'NNRec':#shape (#i,#j,#timepoints,#datasets_to_fit)=(10,10,4,2)
#     np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps2.npy',np.full((10,10,4,2),np.nan))


if key == 'eta':  # Shape (#i, #j, #tries, #eta) = (10, 10, 5, 2)
    eta_sweep = [[[[None for _ in range(2)] for _ in range(5)] for _ in range(10)] for _ in range(10)]
    np.save('../data/SpringMassModel/EtaSweep/eta_sweep' + args.nr + '.npy', eta_sweep)

elif key == 'NNRec_eta':  # Shape (#i, #j, #eta, eta/loss) = (10, 10, 2)
    nnrec_eta_sweep = [[[None for _ in range(2)] for _ in range(10)] for _ in range(10)]
    np.save('../data/SpringMassModel/EtaSweep/eta_sweep' + args.nr + '.npy', nnrec_eta_sweep)

elif key == 'NNRec':  # Shape (#i, #j, #timepoints, #datasets_to_fit) = (10, 10, 4, 2)
    nnrec = [[[[None for _ in range(2)] for _ in range(4)] for _ in range(10)] for _ in range(10)]
    np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps2.npy', nnrec)
