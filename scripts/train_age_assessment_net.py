import sys

# General imports
from   pathlib import Path
import yaml

# Machine learning imports
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# Custom imports
from dl_rad_age.agenet import agenet14_3d, agenet18_3d, agenet18_3d_light, agenet34_3d, update_model_with_autoencoder_weights
from dl_rad_age.evaluation import undo_age_rescaling
from dl_rad_age.checkpointing import buffer_metrics_csv, check_train_status, clean_up_metrics, find_previous_checkpoint, set_training_status
from dl_rad_age.dataloading import FAEDataModule, read_yaml
from dl_rad_age.litmodel import LitModel
from dl_rad_age.losses import AgeLoss_1, AgeLoss_2, AgeLoss_3
from dl_rad_age.transforms import get_transforms


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
        AUGMENTATION       = config['AUGMENTATION']
        AUG_LEVEL          = config['AUG_LEVEL']
        RESCALE_AGE        = config['RESCALE_AGE']
        FLAT_BINS          = config['FLAT_BINS']
        SEX_INPUT          = config['SEX_INPUT']
        BONE_CHANNEL       = config['BONE_CHANNEL']
        NUM_TRAIN_WORKERS  = config['NUM_TRAIN_WORKERS']
        NUM_VALID_WORKERS  = config['NUM_VALID_WORKERS']

        NETWORK            = config['NETWORK']
        DROPOUT            = config['DROPOUT']
        LOSS               = config['LOSS']
        LEARNING_RATE      = config['LEARNING_RATE']
        WEIGHT_DECAY       = config['WEIGHT_DECAY']
        LR_SCHEDULING      = config['LR_SCHEDULING']
        PRETRAINED_WEIGHTS = config['PRETRAINED_WEIGHTS']
        MAX_EPOCHS         = config['MAX_EPOCHS']
        EARLY_STOPPING     = config['EARLY_STOPPING']
    else:
        RUN_DESCRIPTION    = 'agenet18_3d'
        VERSION            = 99
        LOGGING_DIR        = 'lightning_logs'

        ANNOTS_TRAIN       = 'metadata/annotations_train.csv'
        ANNOTS_VALID       = 'metadata/annotations_valid.csv'
        
        BATCH_SIZE         = 16
        AUGMENTATION       = False
        AUG_LEVEL          = 1
        RESCALE_AGE        = True
        FLAT_BINS          = False
        SEX_INPUT          = False
        BONE_CHANNEL       = False
        NUM_TRAIN_WORKERS  = 12
        NUM_VALID_WORKERS  = 4
        
        NETWORK            = 'agenet18_3d'
        DROPOUT            = False
        LOSS               = 'mse'
        LEARNING_RATE      = 0.0003
        WEIGHT_DECAY       = 0
        LR_SCHEDULING      = None
        PRETRAINED_WEIGHTS = None
        MAX_EPOCHS         = 500
        EARLY_STOPPING     = MAX_EPOCHS

    # ------------------------------
    # Paths
    # ------------------------------  

    # Construct paths that are required for training, logging and checkpointing
    checkpoint_dir       = '{:s}/{:s}/version_{:d}/checkpoints'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    config_file          = '{:s}/{:s}/version_{:d}/config.yaml'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    learning_rates_csv   = '{:s}/{:s}/version_{:d}/learning_rates.csv'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    logger_dir           = '{:s}/{:s}'.format(LOGGING_DIR, RUN_DESCRIPTION)
    metrics_dir          = '{:s}/{:s}/version_{:d}'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    training_status_file = '{:s}/{:s}/version_{:d}/training_status.log'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)
    val_results_csv      = '{:s}/{:s}/version_{:d}/val_results.csv'.format(LOGGING_DIR, RUN_DESCRIPTION, VERSION)

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

    # Load latest checkpoint from previous run(s), if available
    last_ckpt = find_previous_checkpoint(checkpoint_dir)
    if last_ckpt:
        print('Resume training with checkpoint <{:s}>'.format(last_ckpt))

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
        patience = EARLY_STOPPING,
        mode     = 'min'
    )

    # ------------------------------
    # Data
    # ------------------------------

    # Datamodule
    fae_dm = FAEDataModule(
        annots_train      = ANNOTS_TRAIN,
        annots_valid      = ANNOTS_VALID,
        batch_size        = BATCH_SIZE,
        transforms_train  = get_transforms(augmentation=AUGMENTATION, level=AUG_LEVEL),
        rescale_age       = RESCALE_AGE,
        flat_bins         = FLAT_BINS,
        include_sex       = SEX_INPUT,
        bone_channel      = BONE_CHANNEL,
        num_train_workers = NUM_TRAIN_WORKERS,
        num_valid_workers = NUM_VALID_WORKERS
    )

    fae_dm.prepare_data()
    fae_dm.setup(stage="fit")

    # ------------------------------
    # Training
    # ------------------------------

    # Number of input channels
    n_channels = 1 if not BONE_CHANNEL else 2
    print('Initialize network with <{:d}> input channels'.format(n_channels))

    # Network
    if NETWORK == 'agenet14_3d':
        network = agenet14_3d(use_dropout=DROPOUT, use_sex=SEX_INPUT)
    elif NETWORK == 'agenet18_3d':
        network = agenet18_3d(num_channels=n_channels, use_dropout=DROPOUT, use_sex=SEX_INPUT)
    elif NETWORK == 'agenet18_3d_light':
        network = agenet18_3d_light(use_dropout=DROPOUT, use_sex=SEX_INPUT)
    elif NETWORK == 'agenet34_3d':
        network = agenet34_3d(use_dropout=DROPOUT, use_sex=SEX_INPUT)
    else:
        raise ValueError('Network name <{:s}> not accepted.'.format(NETWORK))
    n_inputs = 1 if not network.use_sex==True else 2

    # Loss function
    if LOSS=='mae':
        loss = nn.L1Loss()
    elif LOSS=='mse':
        loss = nn.MSELoss()
    elif LOSS=='age_loss_1':
        loss = AgeLoss_1()
    elif LOSS=='age_loss_2':
        loss = AgeLoss_2()
    elif LOSS=='age_loss_3':
        loss = AgeLoss_3()
    else:
        raise ValueError('Invalid loss function key {:s}>'.format(LOSS))

    # Set up model
    model = LitModel(
        net=network,
        n_inputs=n_inputs,
        loss=loss,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduling=LR_SCHEDULING,
        metrics_dir=metrics_dir
    )

    # Pretrained weights
    if PRETRAINED_WEIGHTS:
        # Update model weights with pretrained weights
        autoencoder = NETWORK + '_autoencoder'
        model_dict = update_model_with_autoencoder_weights(
                model_dict=model.state_dict(),
                autoencoder=autoencoder,
                pretrained_weights=PRETRAINED_WEIGHTS,
                sex_input=SEX_INPUT
                )

        # Load the new weights into model
        model.load_state_dict(model_dict)

    # Set up tainer
    dataloader_reload_int = 0
    if FLAT_BINS:
        dataloader_reload_int = 5
    
    trainer = Trainer(
        logger                            = [csv_logger, tb_logger],
        callbacks                         = [mcp_best, mcp_last, early_stop],
        devices                           = 1,
        max_epochs                        = MAX_EPOCHS,
        accelerator                       = 'gpu',
        reload_dataloaders_every_n_epochs = dataloader_reload_int
    )

    # Do training (continue from earlier checkpoint if available)
    set_training_status(training_status_file, 'training in progress...')

    if not last_ckpt:
        trainer.fit(model, fae_dm)
    else:
        buffer_metrics_csv(metrics_dir)
        trainer.fit(model, fae_dm, ckpt_path=last_ckpt)

#***********************************************************************************
# !! Everything below will only be executed if there is no timeout during training !!
#***********************************************************************************

        clean_up_metrics(metrics_dir)

    # ------------------------------
    # Validation
    # ------------------------------

    validator = Trainer(
        devices     = 1,
        accelerator = 'gpu',
        logger      = False
        )
    validation_output = validator.validate(model, fae_dm)

    # Calculate validation error
    val_error = float(validation_output[0]['val_error'])
    val_error = undo_age_rescaling(val_error) - (15 * 365.25)
    val_error = val_error / 365.25
    print('Validation error = {:.3f} years'.format(val_error))

    # Save results
    val_header       = 'metric,value\n'
    val_result       = 'val_error,{:.4f}\n'.format(val_error)
    val_result_lines = [val_header, val_result]

    with open(val_results_csv, "w") as csv_file:
        for line in val_result_lines:
            csv_file.write(line)

    # Set training status to complete
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