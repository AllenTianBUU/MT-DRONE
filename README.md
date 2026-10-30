# MT-DRONE

Code for the paper **"Fast Deep Learning Reconstruction of Magnetization Transfer Parameters from Z-Spectra at Ultra-Low-Field"** (submitted to *Magnetic Resonance in Medicine*).

MT-DRONE is a small neural network that estimates four quantitative magnetization transfer (qMT) parameters, **R1a, M0b, R and T2b**, from Z-spectra and selective inversion-recovery (SelIR) data acquired at 6.5 mT. The network is trained entirely on simulated data, so no measured data is needed to train it.

There are two models, one script each:

| Script | Model | Input |
|---|---|---|
| `mt_drone.py` | Steady-state (SS) | Z-spectra at 5 B1 amplitudes × 52 offsets + 24-point SelIR |
| `mt_drone_time_dependent.py` | Time-dependent (TD) | Z-spectra at 1 B1 (500 Hz) × 38 offsets × 4 saturation times + 24-point SelIR |

## 1. Install

You need Python 3.10 or newer.

```bash
git clone https://github.com/AllenTianBUU/MT-DRONE.git
cd MT-DRONE
pip install -r requirements.txt
```

No GPU is needed. Everything in the paper was trained on a laptop CPU (Apple M1 Pro).

You need about **8 GB of free RAM**, because the full 1.25M-sample training set is held in memory. On a smaller machine, lower `N_SAMPLES` in the SETTINGS block. Training still works, but errors will be somewhat higher.

## 2. Run

```bash
python mt_drone.py                    # steady-state model    (~30 min on an M1 Pro)
python mt_drone_time_dependent.py     # time-dependent model  (a few minutes)
```

Each script runs the full pipeline on its own:

1. Simulates 1.25 million training examples (50,000 parameter sets × 25 noise/nuisance variations).
2. Trains the network.
3. Prints the mean % error on a held-out synthetic test set (10% of the data).
4. Saves plots and a summary to `result_history/runN/` (SS) or `result_history_timedependent/runN/` (TD).

All settings used in the paper are at the top of each script, in the **SETTINGS** block. Nothing else needs to change.

## 3. Expected results

Synthetic test set, mean absolute % error (from the paper):

| Model | R1a | M0b | R | T2b | Average |
|---|---|---|---|---|---|
| Steady-state | 1.19% | 9.96% | 14.21% | 8.66% | 8.51% |
| Time-dependent | 1.59% | 19.63% | 23.86% | 5.52% | 12.65% |

The random training data and network initialization differ from run to run, so your numbers will be close to these but not identical.

## 4. Experimental data (optional)

The phantom measurements used for experimental validation are **not included** in this repository. They are available from the authors on request. Without them, the scripts skip the experimental validation step and still train and report synthetic test errors.

If you have the data, place it next to the scripts like this and rerun:

```
experimental_data/
    zspec_data_MTF_Gelatin.mat
    selIR_data_MTF_Gelatin.mat
    zspec_TRES_250204.mat
    selIR_TRES.mat
experimental_data_time_dependent/          # needed for the TD model only
    Zspec_TsatVary_MTF10_260428.mat
    Zspec_TsatVary_MTF20_260428.mat
    ZspecTsatVary_MTF30.mat
    ZspecTsatVary_MTF40.mat
    ZspecTsatVary_MTF50pc_pH4_shorterTsats.mat
    ZspecTsatVary_MTF100.mat
```

The scripts then also compare the network's predictions to the conventionally fitted values for each phantom. The paper reports these mean experimental errors: **10.7%** (SS) and **36.0%** (TD).

## 5. Using a different acquisition protocol

The networks are tied to the acquisition protocol they were trained on (offset frequencies, B1 amplitudes, saturation times, inversion times). To use your own protocol, edit those arrays in the SETTINGS block and retrain.

## Related code

Conventional non-linear least-squares fitting used for the ground truth: https://github.com/MattRosenLab/low-field-quantitative-mt

## Citation

Citation will be added once the paper is published.
