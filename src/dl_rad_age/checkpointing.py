import csv
import numpy as np
import os
from   pathlib import Path
from   typing import Optional


def set_training_status(training_status_file: str, status: str) -> None:
    # Check if path to file exists
    Path(training_status_file).parent.mkdir(parents=True, exist_ok=True)
    # Write file
    with open(training_status_file, 'w') as f:
        f.write(status)


def check_train_status(training_status_file: str) -> bool:
    # Check if training status file exists
    if Path(training_status_file).is_file():
        print('Attempt to resume previous training...')

        # Read training status file
        with open(training_status_file) as training_status:
            training_status = training_status.readlines()
    
        # Check if file contains only one line, otherwise it might be corrupted
        if len(training_status)==1:

            # Training in progress?
            if training_status[0]=='training in progress...':
                return True

            # Training complete?
            elif training_status[0]=='training complete':
                print('Training was already finished. End script.')
                return False

            else:
                raise ValueError('Expected training status to be "training in progress..." or "training complete". Got "{:d}".'.format(training_status[0]))
        else: 
            raise ValueError('Training status file has too many lines. Expected 1, got {:d}.'.format(len(training_status)))
    # If there is no training status file, the training has not started yet. Create status file.
    else:
        print('Begin new training...')

        # Set training status
        training_status = 'training in progress...'
        print('Write "{:s}": {}'.format(training_status_file, training_status))
        set_training_status(training_status_file, training_status)

    return True


def find_previous_checkpoint(ckpt_dir: str, mode: str = 'last', verbose: bool = True) -> Optional[str]:
    # Verify mode argument
    if not mode in ['last', 'best']:
        raise ValueError('Bad value for argument <mode>')

    # Look for existing checkpoints
    if verbose:
        print('Looking for existing checkpoints in <{:s}>...'.format(ckpt_dir))
    ckpt_path = Path(ckpt_dir)

    if ckpt_path.is_dir():
        
        # Collect checkpoints
        checkpoints = [str(x) for x in ckpt_path.iterdir() if x.is_file() and str(x).endswith('.ckpt')]
        if len(checkpoints)==0:
            raise ValueError('No checkpoints')
        elif verbose:
            print('Found the following checkpoints: {}'.format(checkpoints))

        if mode=='last':
            last_ckpts = [x for x in checkpoints if x.split('/')[-1].startswith(mode)]
            epochs     = [x.split('last-epoch=')[-1].split('-')[0] for x in last_ckpts]
            epochs     = np.asarray([int(x) for x in epochs])

            # Find latest epoch
            idx_latest = np.argmax(epochs)

            # Get latest checkpoint
            last_ckpt = last_ckpts[idx_latest]

            return last_ckpt
        
        elif mode=='best':
            # Find checkpoint that starts with 'best'
            best_ckpt = [x for x in checkpoints if x.split('/')[-1].startswith(mode)]
            if len(best_ckpt)!=1:
                raise ValueError('Wrong number of "best" checkpoints. Expected 1, got {:d}: {}.'.format(len(best_ckpt), best_ckpt))
            
            best_ckpt = best_ckpt[0]
            if verbose:
                print('Return checkpoint <{:s}>'.format(best_ckpt))
            return best_ckpt
    else:
        if verbose:
            print('Checkpoint directory does not exist yet. Return <None>.')
        return None

def read_metrics_csv(metrics_csv: str) -> list:
    metrics = []
    with open(metrics_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            metrics.append(row)
    return metrics

def buffer_metrics_csv(metrics_dir: str) -> None:
    metrics_csv = metrics_dir + '/metrics.csv'
    metrics_tmp = metrics_dir + '/metrics_tmp.csv'
    
    if Path(metrics_tmp).is_file():
        # Read previous <metrics.csv>
        if Path(metrics_csv).is_file():
            curent_metrics = read_metrics_csv(metrics_csv)
        else:
            raise ValueError('File <> should exist.'.format(metrics_csv))

        # Read previous <metrics_tmp.csv>
        temp_metrics = read_metrics_csv(metrics_tmp)

        # Write old metrics, followed by current metrics
        with open(metrics_tmp, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in temp_metrics:
                writer.writerow(row)
            for row in curent_metrics[1:]: # Skip header
                writer.writerow(row)
    else:
        # Rename <metrics.csv> to <metrics_tmp.csv>
        os.rename(metrics_csv, metrics_tmp)

def clean_up_metrics(metrics_dir: str) -> None:
    metrics_csv = metrics_dir + '/metrics.csv'
    metrics_tmp = metrics_dir + '/metrics_tmp.csv'

    # Read previous <metrics.csv>
    if Path(metrics_csv).is_file():
        curent_metrics = read_metrics_csv(metrics_csv)
    else:
        raise ValueError('File <> should exist.'.format(metrics_csv))

    # Read previous <metrics_tmp.csv>
    if Path(metrics_tmp).is_file():
        temp_metrics = read_metrics_csv(metrics_tmp)
    else:
        raise ValueError('File <> should exist.'.format(metrics_tmp))

    # Write old metrics, followed by current metrics
    with open(metrics_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in temp_metrics:
            writer.writerow(row)
        for row in curent_metrics[1:]: # Skip header
            writer.writerow(row)

    # Delete <metrics_tmp.csv>, it is no longer needed
    os.remove(metrics_tmp)