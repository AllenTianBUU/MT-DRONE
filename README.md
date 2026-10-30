# MT-DRONE

Code for the paper **"Fast Deep Learning Reconstruction of Magnetization Transfer Parameters from Z-Spectra at Ultra-Low-Field"**

MT-DRONE is a small neural network that estimates four quantitative magnetization transfer (qMT) parameters, **R1a, M0b, R and T2b**, from Z-spectra and selective inversion-recovery (SelIR) data acquired at 6.5 mT. The network is trained entirely on simulated data, so no measured data is needed to train it.

There are two models, one script each:

| Script | Model | Input |
|---|---|---|
| `mt_drone.py` | Steady-state (SS) | Z-spectra at 5 B1 amplitudes × 52 offsets + 24-point SelIR |
| `mt_drone_time_dependent.py` | Time-dependent (TD) | Z-spectra at 1 B1 (500 Hz) × 38 offsets × 4 saturation times + 24-point SelIR |

## 1. Install

You need Python 3.10 or newer.

```bash
git clone <this-repo-url>
cd MT-DRONE
pip install -r requirements.txt
```

Everything in the paper was trained on a laptop CPU (Apple M1 Pro). 8 GB of RAM is recommended. A dedicated GPU will run the experiment faster, but given the already short training time a GPU is not needed. 

## 2. Run

```bash
python mt_drone.py                    # steady-state model    (~30 min on an M1 Pro)
python mt_drone_time_dependent.py     # time-dependent model  (a few minutes)
```

Each script runs the full pipeline on its own:

1. Simulates 1.25 million (or chosen number of simulated data) of training examples.
2. Trains the network.
3. Prints the mean % error on a held-out synthetic test set (10% of the data).
4. Saves plots and a summary to `result_history/runN/` (SS) or `result_history_timedependent/runN/` (TD).

## 3. Limitation

The networks are tied to the acquisition protocol they were trained on (offset frequencies, B1 amplitudes, saturation times, inversion times). To use your own protocol, edit those arrays in the settings block and retrain.

## Related code

Conventional non-linear least-squares fitting used for the ground truth: https://github.com/MattRosenLab/low-field-quantitative-mt

## Questions

Please reach out to the authors via email. 
