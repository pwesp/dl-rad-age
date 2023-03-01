import numpy as np
import pandas as pd
from   pathlib import Path
import random
import SimpleITK as sitk
from   dl_rad_age.offline_preprocessing import load_nifti, preprocessing



random.seed(0)
np.random.seed(0)



ANNOTATIONS_CSV       = 'metadata/annotations.csv'
BAD_NIFTIS_CSV        = 'metadata/bad_niftis.csv'
IMAGE_INFO_CSV        = 'metadata/image_info.csv'
OUT_DIR               = 'data/preprocessed'
SOI_LOCALIZATIONS_CSV = 'data/soi_localizations.csv'



df_info = pd.read_csv(IMAGE_INFO_CSV)



print('Check if output directory "{:s}" exists...'.format(OUT_DIR))
if Path(OUT_DIR).is_dir():
    print('\t{:s} found'.format(OUT_DIR))
else:
    print('\t"{:s}" not found. Create directory...'.format(OUT_DIR))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print('\t"{:s}" created'.format(OUT_DIR))
    

    
annot_strings = []
bad_niftis    = []
for i, row in enumerate(df_info.iterrows()):
    
    # Get row data
    row_data = row[1]
    
    patient_id = row_data['patient_pseudonym']
    study_id   = row_data['study_pseudonym']
    series_id  = row_data['series_pseudonym']
    loc_x      = row_data['x']
    loc_y      = row_data['y']
    loc_z      = row_data['slice']
    image_file = row_data['image_file']
    age        = row_data['age_at_acquisition']
    sex        = row_data['sex']
    
    print('\nLoad image:')
    print('\tPatient = {:d}, Study = {:d}, Series = {:d}'.format(patient_id, study_id, series_id))
    print('\tImage file = <{:s}>'.format(image_file))
    
    # Load image and convert to numpy array
    image_array, scaling_factor = load_nifti(image_file, resample=True, return_scaling_factor=True)
    
    # Skip images based on the return value of load_nifti()
    if isinstance(image_array, int) and isinstance(scaling_factor, int):
        bad_niftis.append(image_file + '\n')
        print('WARNING: Encountered bad NIfTI. Skip this image.')
        continue     
    if not isinstance(sex, str):
        bad_niftis.append(image_file + '\n')
        print('WARNING: Encountered bad sex value. Skip this image.')
        continue

    # Before continuing, make sure types are correct
    if not isinstance(image_array, np.ndarray):
        raise ValueError('<image_array> is not a numpy array')
    if not isinstance(scaling_factor, np.ndarray):
        raise ValueError('<scaling_factor> is not a numpy array')
    
    # Adjust SOI location for resampling
    loc_x = np.rint(loc_x * scaling_factor[0]).astype(np.int32)
    loc_y = np.rint(loc_y * scaling_factor[1]).astype(np.int32)
    loc_z = np.rint(loc_z * scaling_factor[2]).astype(np.int32)   

    # Apply preprocessing
    print('\nApply preprocessing...')
    image_array = preprocessing(image_array, loc_x, loc_y, loc_z+5, x_length=112, y_length=60, z_length=80)
    if type(image_array) == int:
        if image_array == -1:
            print('ERROR: Something went wrong for this image! Skip image.')
            continue

    # Save numpy array (make sure directory exists)
    preprocessed_image_file = OUT_DIR + '/ae_' + str(patient_id) + '_' + str(study_id) + '_' + str(series_id) + '.npy'
    print('Save image to "{:s}"'.format(preprocessed_image_file))
    np.save(preprocessed_image_file, image_array)
    
    annot_string = preprocessed_image_file + ',' + str(int(age)) + ',' + sex + '\n'
    annot_strings.append(annot_string)

annot_strings = np.asarray(annot_strings)



# Write 'csv' file with all detections
header = 'image,age,sex\n'

with open(ANNOTATIONS_CSV, "w") as csv_file:
    csv_file.write(header)
    for line in annot_strings:
        csv_file.write(line)
        
with open(BAD_NIFTIS_CSV, "w") as csv_file:
    for line in bad_niftis:
        csv_file.write(line)