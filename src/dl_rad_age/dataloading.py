# Custom PyTorch datasets and dataloader wrapped in a custom PyTorch Lightning datamodule
import torch
from   torch.utils.data import DataLoader, Dataset

from   pytorch_lightning import LightningDataModule

import numpy as np
import pandas as pd
import SimpleITK as sitk
from   dl_rad_age.transforms import get_transforms
import torchio as tio
from   typing import Optional
import yaml


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def read_yaml(filepath: str) -> dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))

def hu_to_norm_voxel(hu_value: float) -> float:
    """
    Converts HU values to normalized voxel values.
    Assumptions:
        - CT Window = 500/1500
        - Min-max normalization into value range (-1,1)
    """
    return ((hu_value + 250.0) / 750.0) - 1.0

def norm_voxel_to_hu(norm_voxel: float) -> float:
    """
    Converts HU values to normalized voxel values.
    Assumptions:
        - CT Window = 500/1500
        - Min-max normalization into value range (-1,1)
    """
    return (norm_voxel + 1.0) * 750.0 - 250.0

# --------------------------------------------------
# (PyTorch) Datasets
# --------------------------------------------------

class CTDataset(Dataset):
    """
    CT dataset.

    Attributes
    ----------
        annotation_file : string
            Path to a csv file with annotations
        transforms : list (optional)
            Optional list of transforms to be applied to a sample
    """

    def __init__(
            self,
            annotation_file: str,
            batch_size: int,
            include_sex: bool = False,
            use_full_ct: bool = True,
            cropping: str = 'random',
            transforms_input: Optional[tio.Compose] = None,
            transforms_target: Optional[tio.Compose] = None,
            relative_paths: bool = False
    ) -> None:
        super().__init__()

        self.annotation_file   = annotation_file
        self.batch_size        = batch_size
        self.include_sex       = include_sex
        self.use_full_ct       = use_full_ct
        self.cropping          = cropping
        self.transforms_input  = transforms_input
        self.transforms_target = transforms_target
        self.relative_paths    = relative_paths

        self.annotations = pd.read_csv(annotation_file)
        if self.use_full_ct:
            self.annotations['image'] = self._use_original_images(self.annotations['image'].to_list())

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> tuple:
        """
        Typically, a Dataset returns a single sample. However, this Dataset will return a batch of samples.
        The corresponding DataLoader is adjusted accordingly, by providing a 'collate_fn' function.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filepath
        image_file = str(self.annotations.loc[idx, 'image'])

        # Adjust filepath if necessary
        if self.relative_paths:
            image_file = '../' + image_file

        # Load image from disk
        try:
            if self.use_full_ct:
                image = sitk.ReadImage(image_file)
                image = sitk.GetArrayFromImage(image)
                image = image.astype(np.float32)
            else:
                image = np.load(image_file)
                image = image.astype(np.float32)
        except:
            raise ValueError('Cannot load image <{:s}>'.format(image_file))

        # Get image dimensions
        x, y, z = image.shape[0], image.shape[1], image.shape[2]

        # Prepare cropping
        patch_size = 112

        if self.cropping == 'random':
            rng = np.random.default_rng()
        elif self.cropping == 'fixed':
            rng = np.random.default_rng(seed=0)
        else:
            raise ValueError('"cropping" has to be set to "random" or "fixed", got "{:s}"'.format(self.cropping))

        # Fill batch with patches from the same image to increase performance
        X_batch = []
        y_batch = []

        for i in range(self.batch_size):

            x0, x1 = 0, x
            y0, y1 = 0, y
            z0, z1 = 0, z

            # Generate patch position
            if x > patch_size:
                upper_x0_bound = x - patch_size
                x0 = rng.integers(low=0, high=upper_x0_bound, endpoint=True)
                x1 = x0 + patch_size
            if y > patch_size:
                upper_y0_bound = y - patch_size
                y0 = rng.integers(low=0, high=upper_y0_bound, endpoint=True)
                y1 = y0 + patch_size
            if z > patch_size:
                upper_z0_bound = z - patch_size
                z0 = rng.integers(low=0, high=upper_z0_bound, endpoint=True)
                z1 = z0 + patch_size

            # Do cropping
            patch = image[x0:x1, y0:y1, z0:z1]

            # Prepare dimensions and set up tensors
            patch = np.expand_dims(patch, 0)  # Add color channel dimension
            X_patch = torch.from_numpy(patch)
            y_patch = torch.from_numpy(patch)

            # Apply transforms to input and target image
            if self.transforms_input:
                X_patch = self.transforms_input(X_patch)
            if self.transforms_target:
                y_patch = self.transforms_target(y_patch)

            # Make sure patches are PyTorch tensors
            if type(X_patch) is not torch.Tensor:
                raise ValueError('<X_patch> is not of type <torch.Tensor>')
            if type(y_patch) is not torch.Tensor:
                raise ValueError('<y_patch> is not of type <torch.Tensor>')

            # Add batch dimension. The DataLoader will use this dimension
            # to concatenate all patches inside the batch list
            X_patch = torch.unsqueeze(X_patch, dim=0)
            y_patch = torch.unsqueeze(y_patch, dim=0)

            # Add patch to batch list
            X_batch.append(X_patch)
            y_batch.append(y_patch)

        # Fill metadata batch (always the same because the image is also the same)
        z_batch = []
        if self.include_sex:

            sex = self.annotations.loc[idx, 'sex']

            if sex == 'M':
                sex = torch.tensor([0.0], dtype=torch.float32)
            elif sex == 'F':
                sex = torch.tensor([1.0], dtype=torch.float32)
            else:
                raise ValueError('Sex must be "M" or "F". Got <{}>.'.format(sex))

            # Add batch dimension. The DataLoader will use this dimension
            # to concatenate all patches inside the batch list
            sex = torch.unsqueeze(sex, dim=0)

            z_batch = [sex]*self.batch_size

        if not self.include_sex:
            return X_batch, y_batch
        else:
            return X_batch, y_batch, z_batch

    def _use_original_images(self, image_files) -> list:
        """Replace filepaths for preprocessed images with original images."""
        new_image_files = []

        for old_filename in image_files:

            # Extract info
            patient = old_filename.split('_')[1]
            study = old_filename.split('_')[2]
            series = old_filename.split('_')[3].replace('.npy', '')

            # Create new filename
            new_filename = 'datahdd/AgeEstimation5000/images/ae_{:s}/ae_{:s}/ae_{:s}.nii.gz'.format(patient, study, series)

            # Replace old filename with new filename
            new_image_files.append(new_filename)

        return new_image_files


class FAEDataset(Dataset):
    """
    Forensic age estimation dataset.

    Attributes
    ----------
        annotation_file : string
            Path to a csv file with annotations
        transforms : list (optional)
            Optional list of transforms to be applied to a sample
    """

    def __init__(
            self,
            annotation_file: str,
            transforms: Optional[tio.Compose] = None,
            rescale_age: bool = False,
            flat_bins: bool = False,
            include_sex: bool = False,
            bone_channel: bool = False,
            relative_paths: bool = False,
            return_image_file: bool = False
    ) -> None:
        super().__init__()

        self.annotation_file = annotation_file
        self.transforms = transforms
        self.rescale_age = rescale_age
        self.flat_bins = flat_bins
        self.include_sex = include_sex
        self.bone_channel = bone_channel
        self.relative_paths = relative_paths
        self.return_image_file = return_image_file

        self.orig_annotations = pd.read_csv(annotation_file)
        self.annotations      = self.orig_annotations.copy(deep=True)
        if self.flat_bins:
            self.annotations = self._sample_flat_age_distribution(self.annotations)

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and target
        image_file = str(self.annotations.loc[idx, 'image'])
        age = self.annotations.loc[idx, 'age']
        sex = self.annotations.loc[idx, 'sex']

        # Adjust filepath if necessary
        if self.relative_paths:
            image_file = '../' + image_file

        # Load image from disk
        image = np.load(image_file).astype(np.float32)
        image = np.expand_dims(image, 0)  # Add channel dimension at the front

        # Turn numpy array into torch tensor
        image = torch.from_numpy(image)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        # Create bone segmentation channel based on HU values
        if self.bone_channel:
            hu_threshold = -0.267 # = 300 HU
            bone_seg = torch.where(image > hu_threshold, 1.0, 0.0)

            # Add bone segmentation as second input channel
            image = torch.cat((image, bone_seg), dim=0)

        # Make sure age is a float
        if type(age) is not np.int64:
            raise ValueError('Type of <age> is expected to be <np.int64>, got <{}> instead.'.format(type(age)))

        # Prepare age
        if self.rescale_age:
            age = self._rescale_age(age)
        age = torch.tensor(age, dtype=torch.float32)

        if not self.include_sex:
            if not self.return_image_file:
                return image, age
            else:
                return image, age, image_file
        else:
            # Convert sex from string to float
            if sex == 'M':
                sex = np.float32(0)
            elif sex == 'F':
                sex = np.float32(1)
            else:
                raise ValueError('Value for <sex> is expected to be <M> or <F>, got {} instead.'.format(sex))

            # Prepare sex
            sex = np.expand_dims(sex, 0)  # Add batch dimension at the front
            sex = torch.from_numpy(sex)

            if not self.return_image_file:
                return image, age, sex
            else:
                return image, age, sex, image_file          

    def _rescale_age(self, age: float) -> float:
        """
        Rescale age from the range of [5478 days, 10958 days],
        which is 15 - 30 years, into the range [0.0,1.0] for network training.
        """
        # Lower age bound = 15 years, upper age bound = 30 years
        lower_bound = 5478.0  # = 15 x 365.25
        upper_bound = 10958.0  # = 30 x 365.25

        # Rescale age into [0,1]
        age = (age - lower_bound) / (upper_bound - lower_bound)

        return age

    def _sample_flat_age_distribution(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Samples a flat distribution

        Note: Sampling is done with replacement and therefore alters the original data distribution.
        """
        # Define age bins (a column for sex bins already exists)
        age_bins = np.linspace(start=15.0, stop=30.0, num=16, endpoint=True, dtype=np.int32)

        # Add column with age in years
        annotations['age_years'] = annotations['age'].to_numpy() / 365.25

        # Bin data based on age
        annotations['age_bin'] = np.digitize(annotations['age_years'].to_numpy(), age_bins)

        # Identify number of samples to be drawn per bin, i.e. per age and sex
        _, counts = np.unique(annotations['age_bin'].to_numpy(), return_counts=True)
        n_samples_per_bin = int(np.max(counts)/2.0)

        # Sample a selected number of times from each bin
        annotations = annotations.groupby(['age_bin', 'sex']).apply(lambda x: x.sample(n_samples_per_bin, replace=True)).reset_index(drop=True)

        return annotations

    def reset_annotations(self) -> None:
        if self.flat_bins:
            self.annotations = self._sample_flat_age_distribution(self.orig_annotations)
        else:
            self.annotations = self.orig_annotations

    
# --------------------------------------------------
# Pytorch Lightning datamodules
# --------------------------------------------------

class CTDataModule(LightningDataModule):
    def __init__(
            self,
            annots_train: str,
            annots_valid: str,
            batch_size: int = 1,
            include_sex: bool = False,
            cropping: str = 'random',
            transforms_input: Optional[tio.Compose] = None,
            transforms_target: Optional[tio.Compose] = None,
            num_train_workers = 12,
            num_valid_workers = 4,
            relative_paths = False
    ) -> None:
        super().__init__()

        self.annots_train      = annots_train
        self.annots_valid      = annots_valid
        self.batch_size        = batch_size
        self.include_sex       = include_sex
        self.cropping          = cropping
        self.transforms_input  = transforms_input
        self.transforms_target = transforms_target
        self.num_train_workers = num_train_workers
        self.num_valid_workers = num_valid_workers
        self.relative_paths    = relative_paths

    # Loading the data (performed by all GPUs)
    def setup(self, stage: str) -> None:

        if stage == 'fit' or 'validate':
            self.dataset_train = CTDataset(annotation_file=self.annots_train,
                                           batch_size=self.batch_size,
                                           include_sex=self.include_sex,
                                           cropping=self.cropping,
                                           transforms_input=self.transforms_input,
                                           transforms_target=self.transforms_target,
                                           relative_paths=self.relative_paths)

            self.dataset_val = CTDataset(annotation_file=self.annots_valid,
                                         batch_size=self.batch_size,
                                         include_sex=self.include_sex,
                                         cropping=self.cropping,
                                         transforms_input=self.transforms_input,
                                         transforms_target=self.transforms_target,
                                         relative_paths=self.relative_paths)

        else:
            print('WARNING: This DataModule is for training and validation only and not for testing!')

    def collate_fn(self, data):

        X_batch = data[0][0]
        y_batch = data[0][1]

        X_batch = torch.cat(X_batch)
        y_batch = torch.cat(y_batch)

        if not self.include_sex:
            return X_batch, y_batch
        else:
            z_batch = data[0][2]
            z_batch = torch.cat(z_batch)
            return X_batch, y_batch, z_batch

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=1,
                          shuffle=True,
                          num_workers=self.num_train_workers,
                          collate_fn=self.collate_fn,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_valid_workers,
                          collate_fn=self.collate_fn,
                          persistent_workers=True)


class FAEDataModule(LightningDataModule):
    """
    Custom PyTorch Lightning Module for training and validation dataloading.
    """
    def __init__(
            self,
            annots_train: str,
            annots_valid: str,
            batch_size: int = 8,
            transforms_train: Optional[tio.Compose] = get_transforms(augmentation=False),
            rescale_age: bool = False,
            flat_bins: bool = False,
            include_sex: bool = False,
            bone_channel: bool = False,
            relative_paths: bool = False,
            num_train_workers: int = 8,
            num_valid_workers: int = 4
    ) -> None:
        super().__init__()

        self.annots_train = annots_train
        self.annots_valid = annots_valid
        self.batch_size = batch_size
        self.transforms_train = transforms_train
        self.rescale_age = rescale_age
        self.flat_bins = flat_bins
        self.include_sex = include_sex
        self.bone_channel = bone_channel
        self.relative_paths = relative_paths
        self.num_train_workers = num_train_workers
        self.num_valid_workers = num_valid_workers

    # Loading the data (performed by all GPUs)
    def setup(self, stage: str) -> None:

        if stage == 'fit' or 'validate':
            self.dataset_train = FAEDataset(
                annotation_file=self.annots_train,
                transforms=self.transforms_train,
                rescale_age=self.rescale_age,
                flat_bins=self.flat_bins,
                include_sex=self.include_sex,
                bone_channel=self.bone_channel,
                relative_paths=self.relative_paths
            )
            self.dataset_valid = FAEDataset(
                annotation_file=self.annots_valid,
                transforms=get_transforms(augmentation=False),
                rescale_age=self.rescale_age,
                include_sex=self.include_sex,
                bone_channel=self.bone_channel,
                relative_paths=self.relative_paths
            )
        else:
            raise ValueError('This Datamodule only supports stages "fit" and "validate".')

    def train_dataloader(self):
        if self.flat_bins:
            self.dataset_train.reset_annotations()
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_train_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_valid_workers
        )


class FAETestDataModule(LightningDataModule):
    """
    Custom PyTorch Lightning Module for testing dataloading.
    """
    def __init__(
        self,
        annots: str,
        rescale_age: bool = False,
        include_sex: bool = False,
        bone_channel: bool = False,
        relative_paths: bool = False,
        return_image_file: bool = False,
        num_workers: int = 8
    ) -> None:
        super().__init__()

        self.annots = annots
        self.rescale_age = rescale_age
        self.include_sex = include_sex
        self.bone_channel = bone_channel
        self.num_workers = num_workers
        self.relative_paths = relative_paths
        self.return_image_file = return_image_file

        self.batch_size = 1

    # Loading the data (performed by all GPUs)
    def setup(self, stage: str) -> None:

        if stage == 'test':
            self.dataset = FAEDataset(
                annotation_file=self.annots,
                transforms=get_transforms(augmentation=False),
                rescale_age=self.rescale_age,
                include_sex=self.include_sex,
                bone_channel=self.bone_channel,
                relative_paths=self.relative_paths,
                return_image_file=self.return_image_file
            )
        else:
            raise ValueError('This Datamodule only supports stage "test".')

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
