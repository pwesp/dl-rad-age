import nibabel as nib
import numpy as np
from   pathlib import Path
import SimpleITK as sitk
from   skimage.color import gray2rgb
from   typing import List, Union


def load_nifti(nifti_path, resample=True, return_scaling_factor=True):
    """
    Loads NIfTI image from directory and returns it as numpy array.
    """
    try:
        # Read NIfTI image
        sitk_image = sitk.ReadImage(nifti_path)
    except:
        print('WARNING: SITK failed to read <{:s}>. Return -1 ...'.format(nifti_path))
        return -1, -1
    
    # Resample image
    if resample:
        sitk_image, scaling_factor = resample_image(sitk_image)

    # Turn SITK image into numpy array
    image_array = sitk.GetArrayFromImage(sitk_image)
    
    if return_scaling_factor:
        return image_array, scaling_factor
    
    return image_array



def load_dicom(dicom_path, resample: bool = False, return_scaling_factor: bool = False, verbose: bool = False):
    """
    Loads DICOM image from directory and returns it as numpy array.
    """
    # Convert path to Pathlib object
    dicom_path = Path(dicom_path)
    
    # Collect info from reference DICOM
    ref_image_filename = [str(x.absolute()) for x in Path(dicom_path).iterdir()][0] 
    single_file_reader = sitk.ImageFileReader()
    single_file_reader.SetFileName(ref_image_filename)
    single_file_reader.LoadPrivateTagsOn()
    single_file_reader.ReadImageInformation()
    
    if verbose:
        for k in single_file_reader.GetMetaDataKeys():
            v = single_file_reader.GetMetaData(k)
            print("({0}) = = \"{1}\"".format(k, v))

    # Prepare reading DICOM series
    dicom_path_str      = str(dicom_path.absolute())
    image_series_reader = sitk.ImageSeriesReader()
    dicom_names         = image_series_reader.GetGDCMSeriesFileNames(dicom_path_str) 
    image_series_reader.SetFileNames(dicom_names)

    # Read DICOM series
    sitk_image = image_series_reader.Execute()

    # Resample image
    if resample:
        sitk_image, scaling_factor = resample_image(sitk_image)

    # Turn SITK image into numpy array
    image_array = sitk.GetArrayFromImage(sitk_image)
    
    if return_scaling_factor:
        return image_array, scaling_factor
    
    return image_array



def threshold_ct(image, hu_low=-200, hu_high=600):
    image[image<hu_low]  = hu_low
    image[image>hu_high] = hu_high
    return image



def write_jpg(image_array, jpg_filename, verbose=False, visualize=False):
    
    # Rescale into [0.0, 255.0]
    image_array = np.subtract(image_array, np.min(image_array))
    if np.max(image_array) > 0.0:
        image_array = np.multiply(image_array, 255.0/np.max(image_array))
    
    # Turn into RGB
    image_array = gray2rgb(image_array)
    image_array = np.rint(image_array).astype(np.uint8)
    
    # Visualize array
    if visualize:
        plt.figure(figsize=(6,6))
        plt.imshow(image_array)
        plt.show()
        plt.close()
    
    # Convert so SITK Image and save
    sitk_image  = sitk.GetImageFromArray(image_array, isVector=True)
    writer      = sitk.ImageFileWriter()
    writer.SetFileName(jpg_filename)
    
    writer.Execute(sitk_image)
    
    #sitk.WriteImage(sitk_image, jpg_filename) # doppelt? weiß nicht mehr was hier passiert
    
    if verbose:
        print('Saved "{:s}"'.format(jpg_filename))



def resample_image(sitk_image, new_spacing: list = [1.0,1.0,1.0], verbose: bool = False):
    """
    Resample SITK images.
    """
    
    # Get information about the orginal image
    num_dim        = sitk_image.GetDimension()
    orig_pixelid   = sitk_image.GetPixelIDValue()
    orig_origin    = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing   = np.array(sitk_image.GetSpacing())
    orig_size      = np.array(sitk_image.GetSize(), dtype=np.int32)
    
    if verbose:
        print('Orginal image properties:')
        print('\tnum_dim', num_dim)
        print('\torig_pixelid', orig_pixelid)
        print('\torig_origin', orig_origin)
        print('\torig_direction', orig_direction)
        print('\torig_spacing', orig_spacing)
        print('\torig_size', orig_size)
    
    # Calculate scaling factor
    scaling_factor = orig_spacing/new_spacing
    if verbose:
        print('Scaling factor:', scaling_factor)
    
    # Calculate new size
    new_size = orig_size*scaling_factor
    new_size = np.ceil(new_size).astype(np.int32)
    new_size = [int(s) for s in new_size]
    
    # Define resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    # Perform resampling 
    resampled_sitk_image = resampler.Execute(sitk_image)
    
    # Get information about the resampled image
    num_dim        = resampled_sitk_image.GetDimension()
    orig_pixelid   = resampled_sitk_image.GetPixelIDValue()
    orig_origin    = resampled_sitk_image.GetOrigin()
    orig_direction = resampled_sitk_image.GetDirection()
    orig_spacing   = np.array(resampled_sitk_image.GetSpacing())
    orig_size      = np.array(resampled_sitk_image.GetSize(), dtype=np.int32)
    
    if verbose:
        print('Resampled image properties:')
        print('\tnum_dim', num_dim)
        print('\torig_pixelid', orig_pixelid)
        print('\torig_origin', orig_origin)
        print('\torig_direction', orig_direction)
        print('\torig_spacing', orig_spacing)
        print('\torig_size', orig_size)
    
    return resampled_sitk_image, scaling_factor



def preprocessing(image: np.ndarray, x_loc: int, y_loc: int, z_loc: int, x_length: int = 100, y_length: int = 50, z_length: int = 50):
    """
    Performs image preprocessing operations. For now, this means cropping 
    an image to a uniform size around the structure-of-interest (SOI).
    
    The image dimension are expected to be the following: [z,y,x]
        - 'z' selects the axial plane (y,x-plane)
        - 'y' selects the coronal plane (z,x-plane)
        - 'x' selects the sagitta plane (z,y-plane)
    
    Parameters
    ----------
    image : np.ndarray
        CT scan
    x_loc : int
        x-position of SOI
    y_loc : int
        y-position of SOI
    z_loc : int
        z-position of SOI
    x_length : int (default = 80)
        length of the x-axis after cropping around the SOI
    y_length : int (default = 80)
        length of the z-axis after cropping around the SOI
    z_length : int (default = 40)
        length of the z-axis after cropping around the SOI
        
    Returns
    -------
    image : np.ndarray
        Preprocessed CT scan
    """
    # Calculate 'half-widths' (hws) to crop the image to around the 
    # location (loc) of the SOI like [loc-hw:loc+hw] on each axis
    x_hw = int(x_length / 2.0)
    y_hw = int(y_length / 2.0)
    z_hw = int(z_length / 2.0)
    
    # Check if the specified output axis lengths exceed the original image size
    if x_length > image.shape[2]:
        print('ERROR: x_length {:d} to large for axis length {:d}!'.format(x_length, image.shape[2]))
        return -1
    
    if y_length > image.shape[1]:
        print('ERROR: y_length {:d} to large for axis length {:d}!'.format(y_length, image.shape[1]))
        return -1
    
    if z_length > image.shape[0]:
        print('ERROR: z_length {:d} to large for axis length {:d}!'.format(z_length, image.shape[0]))
        return -1
    
    # Check if the 'half-widths' fit around the location of the SOI. 
    # This might not be the case if the SOI is located closely to the 
    # edges of the original image. If not, change the SOI location. 
    # In case the 'half-widths' do not fit, change the SOI location
    # coordinates, such that the image can be cropped around the SOI
    # with the specified axis lengths
    if x_loc-x_hw < 0:
        x_loc += x_loc-x_hw
        print('WARNING: Adjusted x-coordinate of SOI location to {:d}!'.format(x_loc))
            
    if y_loc-y_hw < 0:
        y_loc += y_loc-y_hw
        print('WARNING: Adjusted y-coordinate of SOI location to {:d}!'.format(y_loc))

    if z_loc-z_hw < 0:
        z_loc += z_loc-z_hw
        print('WARNING: Adjusted z-coordinate of SOI location to {:d}!'.format(z_loc))
        
    if x_loc+x_hw > image.shape[2]:
        x_loc -= (x_loc+x_hw) - image.shape[2]
        print('WARNING: Adjusted x-coordinate of SOI location to {:d}!'.format(x_loc))
            
    if y_loc+y_hw > image.shape[1]:
        y_loc -= (y_loc+y_hw) - image.shape[1]
        print('WARNING: Adjusted y-coordinate of SOI location to {:d}!'.format(y_loc))

    if z_loc+z_hw > image.shape[0]:
        z_loc -= (z_loc+z_hw) - image.shape[0]
        print('WARNING: Adjusted z-coordinate of SOI location to {:d}!'.format(z_loc))
    
    # Do cropping
    image = image[z_loc-z_hw:z_loc+z_hw, y_loc-y_hw:y_loc+y_hw, x_loc-x_hw:x_loc+x_hw]
    
    # Sanity check
    if image.shape!=(z_length, y_length, x_length):
        print('ERROR: Image shape is {}. Should be {}!'.format(image.shape, (z_length, y_length, x_length)))
        return -1
    
    return image