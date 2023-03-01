from   pytorch_lightning import Trainer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from   pathlib import Path, PurePath
from   typing import Optional

from   dl_rad_age.agenet import agenet14_3d, agenet18_3d, agenet18_3d_light, agenet34_3d
from   dl_rad_age.checkpointing import find_previous_checkpoint
from   dl_rad_age.dataloading import FAETestDataModule, read_yaml
from   dl_rad_age.litmodel import LitModel


def create_results_folder_name(run: str) -> str:
    """Create name for results folder from run description"""
    results_folder = '{:s}_v{:s}'.format(run.split('/')[0], run.split('/')[1].split('_')[1])
    return results_folder


def filter_runs(runs: list, include: Optional[list] = None, exclude: Optional[list] = None) -> list:

    if include:
        for token in include:
            runs = [x for x in runs if token in x]

    if exclude:
        for token in exclude:
            runs = [x for x in runs if token not in x]

    return runs


def get_runs(logs_dir: str) -> list:
    """
    Get all runs stored in a machine learning logging directory.
    A run consists of a run description and a version number, e.g. <network_XYZ/version_0>
    """
    run_descriptions = [str(x.name) for x in Path(logs_dir).iterdir() if x.is_dir()]
    
    runs = []
    for r in run_descriptions:
        versions = [x.name for x in Path(PurePath(logs_dir, r)).iterdir() if x.is_dir() and str(x.name).startswith('version') and not str(x.name).startswith('.')]
        if versions:
            for v in versions:
                run = str(PurePath(r, v))
                runs.append(run)

    return runs


def get_run_result(run: str, results_dir: str, results_file: str = 'val_results.csv'):

    csv_filepath = PurePath(results_dir, run, results_file)

    df_result = pd.read_csv(csv_filepath)
    result    = df_result.loc[0,'value']
    return result


def get_run_metrics(run: str, results_dir: str, metrics_file: str = 'metrics.csv') -> tuple:

    csv_filepath = PurePath(results_dir, run, metrics_file)

    df_metrics = pd.read_csv(csv_filepath)

    loss     = df_metrics['loss'].to_numpy()
    epoch    = df_metrics['epoch'].to_numpy()
    step     = df_metrics['step'].to_numpy()
    val_loss = df_metrics['val_loss'].to_numpy()

    loss       = loss[df_metrics['loss'].notna()]
    loss_epoch = epoch[df_metrics['loss'].notna()]
    loss_step  = step[df_metrics['loss'].notna()]

    frac_epoch_factor = np.divide(np.max(loss_epoch), np.max(loss_step))
    loss_epoch = np.multiply(loss_step, frac_epoch_factor)

    val_loss       = val_loss[df_metrics['val_loss'].notna()]
    val_loss_epoch = epoch[df_metrics['val_loss'].notna()]

    return loss_epoch, loss, val_loss_epoch, val_loss


def undo_age_rescaling(age: float) -> float:
    
    # Lower age bound = 15 years, upper age bound = 30 years
    lower_bound = 5478.0  # 15 x 365.25
    upper_bound = 10958.0 # 30 x 365.25
    
    # Rescale age back into [0,1]
    age = (age * (upper_bound - lower_bound)) + lower_bound  
    
    return age


def get_model_predictions(
    datamodule: FAETestDataModule,
    model: LitModel,
    undo_scaling: bool = False
    ) -> tuple:

    # Set model to evaluation mode
    model.eval()
    
    # Set up lightning Trainer
    tester = Trainer(
        devices     = 1,
        accelerator = 'gpu',
        logger      = False
        )

    # Do testing
    tester.test(model, datamodule)

    # Get results
    test_results = model.test_results

    # Reformat results
    test_results = np.asarray(test_results)
    test_results = test_results.swapaxes(0,1)

    y_true = test_results[0]
    y_pred = test_results[1]

    # Undo age rescaling
    if undo_scaling:
        y_true = undo_age_rescaling(y_true)
        y_pred = undo_age_rescaling(y_pred)
    else:
        y_pred = undo_age_rescaling(y_pred)
    
    if not datamodule.include_sex and model.n_inputs==1:
        if not model.track_image_files:
            return y_true, y_pred
        else:
            image_files = np.asarray(model.image_files).squeeze()
            return y_true, y_pred, image_files
    elif datamodule.include_sex and model.n_inputs==2:
        sex = test_results[2]
        if not model.track_image_files:
            return y_true, y_pred, sex
        else:
            image_files = np.asarray(model.image_files).squeeze()
            return y_true, y_pred, sex, image_files
    else:
        raise ValueError('Bad number of inputs.')


def evaluate_run(run: str, annots: str, logs_dir: str, results_dir: str, track_image_files: bool = False, make_violin_plot: bool = True) -> None:

    print('run: {:s}'.format(run))
    config_filepath = str(PurePath(logs_dir, run, 'config.yaml'))
    print('\tLoad <{:s}>'.format(config_filepath))
    config          = read_yaml(config_filepath)

    # Check for certain newly added keys (for old configs)
    if not 'BONE_CHANNEL' in config.keys():
        config['BONE_CHANNEL'] = False
    
    # Datamodule
    fae_dm = FAETestDataModule(
        annots            = str(PurePath('..', config['ANNOTS_VALID'])),
        rescale_age       = config['RESCALE_AGE'],
        include_sex       = config['SEX_INPUT'],
        bone_channel      = config['BONE_CHANNEL'],
        relative_paths    = True,
        return_image_file = track_image_files
        )
    fae_dm.prepare_data()
    fae_dm.setup(stage='test')
    
    # Model
    ckpt_dir  = str(PurePath(logs_dir, run, 'checkpoints'))
    prev_ckpt = find_previous_checkpoint(ckpt_dir, verbose=False, mode='best')

    # Number of input channels
    n_channels = 1 if not config['BONE_CHANNEL'] else 2

    if config['NETWORK'] == 'agenet14_3d':
        network = agenet14_3d(use_sex=config['SEX_INPUT'])
    elif config['NETWORK'] == 'agenet18_3d':
        network = agenet18_3d(num_channels=n_channels, use_sex=config['SEX_INPUT'])
    elif config['NETWORK'] == 'agenet18_3d_light':
        network = agenet18_3d_light(use_sex=config['SEX_INPUT'])
    elif config['NETWORK'] == 'agenet34_3d':
        network = agenet34_3d(num_channels=n_channels, use_sex=config['SEX_INPUT'])
    else:
        raise ValueError('Network name <{:s}> not accepted'.format(config['NETWORK']))
    n_inputs = 1 if not network.use_sex==True else 2

    if prev_ckpt:
        model = LitModel.load_from_checkpoint(
            prev_ckpt,
            net=network,
            n_inputs=n_inputs,
            metrics_dir=str(PurePath(logs_dir, run)),
            track_image_files=track_image_files
        )
    else:
        raise ValueError('No previous checkpoint available.')
    
    # Make predictions
    if not config['SEX_INPUT']:
        if not track_image_files:
            y_true, y_pred_dl = get_model_predictions(fae_dm, model, undo_scaling=config['RESCALE_AGE'])
        else:
            y_true, y_pred_dl, image_files = get_model_predictions(fae_dm, model, undo_scaling=config['RESCALE_AGE'])
    else:
        if not track_image_files:
            y_true, y_pred_dl, sex = get_model_predictions(fae_dm, model, undo_scaling=config['RESCALE_AGE'])
        else:
            y_true, y_pred_dl, sex, image_files = get_model_predictions(fae_dm, model, undo_scaling=config['RESCALE_AGE'])

    print('y_true', y_true.shape)
    print('y_pred_dl', y_pred_dl.shape)
    print('sex', sex.shape)
    print('image_files', image_files.shape)
    
    # Evaluation
    y_true_years    = y_true / 365.25
    y_pred_dl_years = y_pred_dl / 365.25
    abs_error_dl    = np.abs(np.subtract(y_true_years, y_pred_dl_years))

    # Save predictions
    if not config['SEX_INPUT']:
        if not track_image_files:
            df_results = pd.DataFrame(
                data = np.array([y_true_years, y_pred_dl_years, abs_error_dl]).swapaxes(0,1),
                columns=['y_true', 'y_pred', 'abs_error']
                )
        else:
            df_results = pd.DataFrame(
                data = np.array([y_true_years, y_pred_dl_years, abs_error_dl, image_files]).swapaxes(0,1),
                columns=['y_true', 'y_pred', 'abs_error', 'image_file']
                )
    else:
        if not track_image_files:
            df_results = pd.DataFrame(
                data = np.array([y_true_years, y_pred_dl_years, abs_error_dl, sex]).swapaxes(0,1),
                columns=['y_true', 'y_pred', 'abs_error', 'sex']
            )
        else:
            df_results = pd.DataFrame(
                data = np.array([y_true_years, y_pred_dl_years, abs_error_dl, sex, image_files]).swapaxes(0,1),
                columns=['y_true', 'y_pred', 'abs_error', 'sex', 'image_file']
            )
    
    # Create directories and filename for results
    results_filepath = str(PurePath(results_dir, create_results_folder_name(run), 'results.csv'))
    print('\tSave <{:s}>'.format(results_filepath))
    df_results.to_csv(results_filepath, index=False)

    # Summary
    mae_dl     = np.mean(abs_error_dl)
    sd_dl      = np.std(abs_error_dl)
    df_results_summary = pd.DataFrame(
        data = np.array([[mae_dl], [sd_dl]]).swapaxes(0,1),
        columns=['mean', 'sd']
        )

    # Create directories and filename for summary results
    results_summary_filepath = str(PurePath(results_dir, create_results_folder_name(run), 'results_summary.csv'))
    print('\tSave <{:s}>'.format(results_summary_filepath))
    df_results_summary.to_csv(results_summary_filepath, index=False)

    # Bin results
    bins          = np.linspace(15, 29, num=8, dtype=np.int32)
    dl_bin_inds   = np.digitize(y_true_years, bins, right=True)
    if not config['SEX_INPUT']:
        dl_mae_binned = [abs_error_dl[dl_bin_inds==i] for i in np.unique(dl_bin_inds)]
    else:
        mae_dl_m = np.mean(abs_error_dl[sex==0.0])
        mae_dl_f = np.mean(abs_error_dl[sex==1.0])
        dl_mae_binned_m = [abs_error_dl[(dl_bin_inds==i) & (sex==0.0)] for i in np.unique(dl_bin_inds)]
        dl_mae_binned_f = [abs_error_dl[(dl_bin_inds==i) & (sex==1.0)] for i in np.unique(dl_bin_inds)]

    # Create directories and filename for violin plot
    violinplot_filepath = PurePath(results_dir, create_results_folder_name(run), 'abs_err_plot.png')
    Path(violinplot_filepath).parent.mkdir(parents=True, exist_ok=True)

    # Create violin plot
    if make_violin_plot:
        if not config['SEX_INPUT']:
            plt.figure(figsize=(8,8))
            vp_dl_parts = plt.violinplot(dl_mae_binned, positions=bins, showmeans=True)
            for part in vp_dl_parts['bodies']:
                part.set_linewidth(2)
                part.set_facecolor('cornflowerblue')
                part.set_edgecolor('black')
            plt.axhline(y=mae_dl, lw=2, c='r', label='MAE = {:.2f}'.format(mae_dl))
            plt.ylim(-0.5,8.5)
            plt.tick_params(labelsize=16)
            plt.xlabel('age / (years)', fontsize=20)
            plt.ylabel('abs err / (years)', fontsize=20)
            plt.legend(fontsize=16)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(16,8))
            vp_dl_m_parts = ax[0].violinplot(dl_mae_binned_m, positions=bins, showmeans=True)
            ax[0].axhline(y=mae_dl_m, lw=2, c='r', label='MAE (male) = {:.2f}'.format(mae_dl_m))
            for part in vp_dl_m_parts['bodies']:
                part.set_linewidth(2)
                part.set_facecolor('cornflowerblue')
                part.set_edgecolor('black')
            vp_dl_f_parts = ax[1].violinplot(dl_mae_binned_f, positions=bins, showmeans=True)
            ax[1].axhline(y=mae_dl_f, lw=2, c='r', label='MAE (female) = {:.2f}'.format(mae_dl_f))
            for part in vp_dl_f_parts['bodies']:
                part.set_linewidth(2)
                part.set_facecolor('cornflowerblue')
                part.set_edgecolor('black')
            for axis in ax:
                axis.set_ylim(-0.5,8.5)
                axis.tick_params(labelsize=16)
                axis.set_xlabel('age / (years)', fontsize=20)
                axis.set_ylabel('abs err / (years)', fontsize=20)
                axis.legend(fontsize=16)
        print('\tSave <{:s}>'.format(str(violinplot_filepath)))
        plt.savefig(violinplot_filepath)
        plt.close()