# dl-rad-age

Deep learning enhanced radiological age assessment

## Prerequisits

## Data preprocessing

Raw data was preprocessed with `scripts/offline_preprocessing_age_estimation.py`

## Model training

### Networks

Individual networks were trained with `scripts/train_age_assessment_net.py`

### Ensemble

The ensemble model consisted of multiple instances of the same network trained using the configurations specified in `configs`.

#### Weights

The weights of the 20 trained networks can be found here:

## Model evaluation

The (ensemble) model was evaluated using the following notebooks

- `calculate_age_assessment_ensemble_results.ipynb`
- `analyze_age_assessment_ensemble_results.ipynb`
- `plot_results_age_assessment.ipynb`

## MISC

- The code was developed using the Python environment inside this [Docker container](https://hub.docker.com/layers/balthasarschachtner/radler/pytorch_v3.1/images/sha256-1720c6706699658c41546fabc6fca809b3fc0e4c61deb2271d2d36f2343c8037?context=repo)
- Computations were performed on a Nvidia DGX with A100 GPUs using SLURM for workload management
