import sys

# General imports
import csv
from   pathlib import Path
import yaml

# Machine learning imports
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# Custom imports
from dl_rad_age.agenet_autencoder import agenet14_3d_autoencoder, agenet18_3d_autoencoder, agenet18_3d_light_autoencoder, agenet34_3d_autoencoder
from dl_rad_age.checkpointing import check_train_status, find_previous_checkpoint, read_metrics_csv, set_training_status, update_metrics_csv
from dl_rad_age.dataloading import CTDataModule, read_yaml
from dl_rad_age.litmodel import LitModel_Autoencoder
from dl_rad_age.transforms import get_autoencoder_transforms


def main(config: dict = {}) -> None:

    # ------------------------------
    # Config
    # ------------------------------

    # Use settings specified in config file if available. Otherwise use default settings.
    if dict:
        RUN_DESCRIPTION    = config['RUN_DESCRIPTION']
        VERSION            = config['VERSION']
        LOGGING_DIR        = config['LOGGING_DIR']

        ANNOTS_TRAIN       = config['ANNOTS_TRAIN']
        ANNOTS_VALID       = config['ANNOTS_VALID']

        BATCH_SIZE         = config['BATCH_SIZE']
        SEX_INPUT          = config['SEX_INPUT']
        NUM_TRAIN_WORKERS  = config['NUM_TRAIN_WORKERS']
        NUM_VALID_WORKERS  = config['NUM_VALID_WORKERS']

        NETWORK            = config['NETWORK']
        MAX_EPOCHS         = config['MAX_EPOCHS']
    else:
        RUN_DESCRIPTION    = 'agenet18_3d_autoencoder'
        VERSION            = 99
        LOGGING_DIR        = 'lightning_logs'

        ANNOTS_TRAIN       = 'metadata/annotations_train.csv'
        ANNOTS_VALID       = 'metadata/annotations_valid.csv'
        
        BATCH_SIZE         = 16
        SEX_INPUT          = False
        NUM_TRAIN_WORKERS  = 12
        NUM_VALID_WORKERS  = 4
        
        NETWORK            = 'agenet18_3d_autoencoder'
        MAX_EPOCHS         = 500

    # ------------------------------
    # Paths
    # ------------------------------  

    # Construct paths that are required for training, logging and checkpointing
    checkpoint_dir       = '{:s}/{:s}/version_{:d}/checkpoints'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    config_file          = '{:s}/{:s}/version_{:d}/config.yaml'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    logger_dir           = '{:s}/{:s}'.format(LOGGING_DIR, RUN_DESCRIPTION)
    metrics_csv          = '{:s}/{:s}/version_{:d}/metrics.csv'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    training_status_file = '{:s}/{:s}/version_{:d}/training_status.log'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)

    # ------------------------------
    # Checkpointing
    # ------------------------------  

    # Save config
    Path(config_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

    # Check training status to see if further training is required
    training_status = check_train_status(training_status_file)
    if training_status == False:
        return None

    # When available, load best checkpoint from previous run
    best_ckpt = find_previous_checkpoint(checkpoint_dir)
    if best_ckpt:
        print('Resume training with checkpoint <{:s}>'.format(best_ckpt))

    # ------------------------------
    # Training status (for DGX)
    # ------------------------------

    # Check training status to see if further training is required
    if Path(training_status_file).is_file():

        with open(training_status_file) as training_status:
            training_status = training_status.readlines()
    
        if len(training_status)==1:
            if training_status[0]=='training complete':
                print('Training was already finished. End script.')
                return None
        else:
            raise ValueError('Training status file has too many lines. Expected 1, got {:d}'.format(len(training_status)))

    # ------------------------------
    # Logging setup
    # ------------------------------

    # Initialize logger
    csv_logger = CSVLogger(
        save_dir = '.',
        name     = logger_dir,
        version  = 'version_{:d}'.format(VERSION)
    )

    tb_logger = TensorBoardLogger(
        save_dir = '.',
        name     = logger_dir,
        version  = 'version_{:d}'.format(VERSION)
    )

    # Initialize checkpoint callbacks
    mcp_best = ModelCheckpoint(
        dirpath                 = checkpoint_dir,
        filename                = 'best-epoch={epoch}-val_loss={val_loss:.2f}',
        monitor                 = 'val_loss',
        save_top_k              = 1,
        auto_insert_metric_name = False
    )

    mcp_last = ModelCheckpoint(
        dirpath                 = checkpoint_dir,
        filename                = 'last-epoch={epoch}-val_loss={val_loss:.2f}',
        auto_insert_metric_name = False
    )

    # Initialize other callbacks
    early_stop = EarlyStopping(
        monitor  = "val_loss",
        patience = 20,
        mode     = 'min'
    )

    # ------------------------------
    # Data
    # ------------------------------

    # Datamodule
    ct_dm = CTDataModule(
        annots_train      = ANNOTS_TRAIN,
        annots_valid      = ANNOTS_VALID,
        batch_size        = BATCH_SIZE,
        include_sex       = SEX_INPUT,
        transforms_input  = get_autoencoder_transforms('input'),
        transforms_target = get_autoencoder_transforms('target'),
        num_train_workers = NUM_TRAIN_WORKERS,
        num_valid_workers = NUM_VALID_WORKERS
    )

    ct_dm.prepare_data()
    ct_dm.setup(stage="fit")

    # ------------------------------
    # Training
    # ------------------------------

    # Network
    if NETWORK == 'agenet14_3d_autoencoder':
        network = agenet14_3d_autoencoder(use_sex=SEX_INPUT)
    elif NETWORK == 'agenet18_3d_autoencoder':
        network = agenet18_3d_autoencoder(use_sex=SEX_INPUT)
    elif NETWORK == 'agenet18_3d_light_autoencoder':
        network = agenet18_3d_light_autoencoder(use_sex=SEX_INPUT)
    elif NETWORK == 'agenet34_3d_autoencoder':
        network = agenet34_3d_autoencoder(use_sex=SEX_INPUT)
    else:
        raise ValueError('Network name <{:s}> not accepted.'.format(NETWORK))
    n_inputs = 1 if not network.use_sex==True else 2

    # Set up model
    model = LitModel_Autoencoder(
        net=network,
        n_inputs=n_inputs
    )
    
    trainer = Trainer(
        logger             = [csv_logger, tb_logger],
        callbacks          = [mcp_best, mcp_last, early_stop],
        devices            = 1,
        max_epochs         = MAX_EPOCHS,
        accelerator        = 'gpu',
        val_check_interval = float(1.0/BATCH_SIZE)
    )

    # Do training (continue from earlier checkpoint if available)
    set_training_status(training_status_file, 'training in progress...')

    if not best_ckpt:
        trainer.fit(model, ct_dm)
    else:
        old_metrics = read_metrics_csv(metrics_csv)
        trainer.fit(model, ct_dm, ckpt_path=best_ckpt)
        update_metrics_csv(metrics_csv, old_metrics)

    set_training_status(training_status_file, 'training complete')

    return None


if __name__ == '__main__':
    
    # Get config file from command line
    if len(sys.argv)==1:
        print('No config file specified. Continue with default settings.')
        config = {}
    elif len(sys.argv)==2:
        print('Load and use settings from config file {:s}'.format(sys.argv[1]))
        config_file = sys.argv[1]
        config      = read_yaml(config_file)
    else:
        print('ERROR: Too many arguments. Expected 1, but {:d} were given. Quit script.'.format(len(sys.argv)-1))
        quit()

    main(config)