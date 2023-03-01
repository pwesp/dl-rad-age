from   torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
from   dl_rad_age.evaluation import undo_age_rescaling



def plot_ct_dataset(dataset: Dataset, batch: int, sample: int, x: int, y: int, z: int) -> None:
    
    if len(dataset[batch]) == 2:
        X_batch, y_batch = dataset[batch]
    elif len(dataset[batch]) == 3:
        X_batch, y_batch, z_batch = dataset[batch]
        
    X_image = X_batch[sample].numpy()
    y_image = y_batch[sample].numpy()
    
    print('X_image: {} [min = {:.1f}, max = {:.1f}]'.format(X_image.shape, np.min(X_image), np.max(X_image)))
    print('y_image: {} [min = {:.1f}, max = {:.1f}]'.format(y_image.shape, np.min(y_image), np.max(y_image)))
    
    if len(dataset[batch]) == 3:
        sex = z_batch[sample].numpy()
        print('sex:    {} [value = {}]'.format(sex.shape, sex))

    X_image = X_image[0,0,:]
    y_image = y_image[0,0,:]
    
    hist_X, hu_edges_X = np.histogram(X_image, bins = 20, range = (-1, 1))
    hist_y, hu_edges_y = np.histogram(y_image, bins = 20, range = (0, 1))
    
    fig, ax = plt.subplots(2, 4, figsize=(36,18))
    
    ax[0,0].imshow(X_image[z,:,:])
    ax[0,0].set_xlabel('x', fontsize=18)
    ax[0,0].set_ylabel('y', fontsize=18)
    ax[0,0].set_title('axial', fontsize=24)
    
    ax[0,1].imshow(X_image[:,y,:])
    ax[0,1].set_xlabel('x', fontsize=18)
    ax[0,1].set_ylabel('z', fontsize=18)
    ax[0,1].set_title('coronal', fontsize=24)
    
    ax[0,2].imshow(X_image[:,:,x])
    ax[0,2].set_xlabel('y', fontsize=18)
    ax[0,2].set_ylabel('z', fontsize=18)
    ax[0,2].set_title('sagittal', fontsize=24)
    
    ax[0,3].bar(hu_edges_X[:-1], hist_X, width=0.08, color='cornflowerblue')
    ax[0,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[0,3].set_ylabel('count', fontsize=18)
    ax[0,3].set_title('intensity distribution', fontsize=24)
    
    ax[1,0].imshow(y_image[z,:,:])
    ax[1,0].set_xlabel('x', fontsize=18)
    ax[1,0].set_ylabel('y', fontsize=18)
    ax[1,0].set_title('axial', fontsize=24)
    
    ax[1,1].imshow(y_image[:,y,:])
    ax[1,1].set_xlabel('x', fontsize=18)
    ax[1,1].set_ylabel('z', fontsize=18)
    ax[1,1].set_title('coronal', fontsize=24)
    
    ax[1,2].imshow(y_image[:,:,x])
    ax[1,2].set_xlabel('y', fontsize=18)
    ax[1,2].set_ylabel('z', fontsize=18)
    ax[1,2].set_title('sagittal', fontsize=24)
    
    ax[1,3].bar(hu_edges_y[:-1], hist_y, width=0.04, color='cornflowerblue')
    ax[1,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[1,3].set_ylabel('count', fontsize=18)
    ax[1,3].set_title('intensity distribution', fontsize=24)
    
    plt.show()

    
    
def plot_ct_datamodule_batch(batch: list, sample: int, x: int, y: int, z: int) -> None:
    
    X_image = batch[0][sample].numpy()
    y_image = batch[1][sample].numpy()
    if len(batch)==3:
        sex = batch[2][sample].numpy()

    
    print('X_image: {} [min = {:.1f}, max = {:.1f}]'.format(X_image.shape, np.min(X_image), np.max(X_image)))
    print('y_image: {} [min = {:.1f}, max = {:.1f}]'.format(y_image.shape, np.min(y_image), np.max(y_image)))
    if len(batch)==3: 
        print('sex:    {} [value = {}]'.format(sex.shape, sex))

    X_image = X_image[0,:]
    y_image = y_image[0,:]
    
    hist_X, hu_edges_X = np.histogram(X_image, bins = 20, range = (-1, 1))
    hist_y, hu_edges_y = np.histogram(y_image, bins = 20, range = (0, 1))
    
    fig, ax = plt.subplots(2, 4, figsize=(36,18))
    
    ax[0,0].imshow(X_image[z,:,:])
    ax[0,0].set_xlabel('x', fontsize=18)
    ax[0,0].set_ylabel('y', fontsize=18)
    ax[0,0].set_title('axial', fontsize=24)
    
    ax[0,1].imshow(X_image[:,y,:])
    ax[0,1].set_xlabel('x', fontsize=18)
    ax[0,1].set_ylabel('z', fontsize=18)
    ax[0,1].set_title('coronal', fontsize=24)
    
    ax[0,2].imshow(X_image[:,:,x])
    ax[0,2].set_xlabel('y', fontsize=18)
    ax[0,2].set_ylabel('z', fontsize=18)
    ax[0,2].set_title('sagittal', fontsize=24)
    
    ax[0,3].bar(hu_edges_X[:-1], hist_X, width=0.08, color='cornflowerblue')
    ax[0,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[0,3].set_ylabel('count', fontsize=18)
    ax[0,3].set_title('intensity distribution', fontsize=24)    

    ax[1,0].imshow(y_image[z,:,:])
    ax[1,0].set_xlabel('x', fontsize=18)
    ax[1,0].set_ylabel('y', fontsize=18)
    ax[1,0].set_title('axial', fontsize=24)
    
    ax[1,1].imshow(y_image[:,y,:])
    ax[1,1].set_xlabel('x', fontsize=18)
    ax[1,1].set_ylabel('z', fontsize=18)
    ax[1,1].set_title('coronal', fontsize=24)
    
    ax[1,2].imshow(y_image[:,:,x])
    ax[1,2].set_xlabel('y', fontsize=18)
    ax[1,2].set_ylabel('z', fontsize=18)
    ax[1,2].set_title('sagittal', fontsize=24)
    
    ax[1,3].bar(hu_edges_y[:-1], hist_y, width=0.04, color='cornflowerblue')
    ax[1,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[1,3].set_ylabel('count', fontsize=18)
    ax[1,3].set_title('intensity distribution', fontsize=24)
    
    plt.show()



def plot_ct_dataset_predictions(model, dataset, batch: int, sample: int, x: int, y: int, z: int) -> None:
    
    if len(dataset[batch])==2:
        X_batch, y_batch = dataset[batch]
    if len(dataset[batch])==3:
        X_batch, y_batch, z_batch = dataset[batch]
        
    X_image = X_batch[sample].numpy()
    y_image = y_batch[sample].numpy()
    if len(dataset[batch])==2:
        y_pred = model(X_batch[sample]).detach().numpy()
    if len(dataset[batch])==3:
        y_pred = model(X_batch[sample], z_batch[sample]).detach().numpy()
        
    print('X_image: {} [min = {:.1f}, max = {:.1f}]'.format(X_image.shape, np.min(X_image), np.max(X_image)))
    print('y_image: {} [min = {:.1f}, max = {:.1f}]'.format(y_image.shape, np.min(y_image), np.max(y_image)))
    print('y_pred:  {} [min = {:.1f}, max = {:.1f}]'.format(y_pred.shape, np.min(y_pred), np.max(y_pred)))
    
    X_image = X_image[0,0,:]
    y_image = y_image[0,0,:]
    y_pred  = y_pred[0,0,:]
    
    hist_y, hu_edges_y           = np.histogram(y_image, bins = 20, range = (0, 1))
    hist_y_pred, hu_edges_y_pred = np.histogram(y_pred, bins = 20, range = (0, 1))
    
    fig, ax = plt.subplots(2, 4, figsize=(28,14))
    
    ax[0,0].imshow(y_image[z,:,:])
    ax[0,0].set_xlabel('x', fontsize=18)
    ax[0,0].set_ylabel('y', fontsize=18)
    ax[0,0].set_title('axial', fontsize=24)
    
    ax[0,1].imshow(y_image[:,y,:])
    ax[0,1].set_xlabel('x', fontsize=18)
    ax[0,1].set_ylabel('z', fontsize=18)
    ax[0,1].set_title('coronal', fontsize=24)
    
    ax[0,2].imshow(y_image[:,:,x])
    ax[0,2].set_xlabel('y', fontsize=18)
    ax[0,2].set_ylabel('z', fontsize=18)
    ax[0,2].set_title('sagittal', fontsize=24)
    
    ax[0,3].bar(hu_edges_y[:-1], hist_y, width=0.08, color='cornflowerblue')
    ax[0,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[0,3].set_ylabel('count', fontsize=18)
    ax[0,3].set_title('intensity distribution', fontsize=24)
    
    ax[1,0].imshow(y_pred[z,:,:])
    ax[1,0].set_xlabel('x', fontsize=18)
    ax[1,0].set_ylabel('y', fontsize=18)
    ax[1,0].set_title('axial', fontsize=24)
    
    ax[1,1].imshow(y_pred[:,y,:])
    ax[1,1].set_xlabel('x', fontsize=18)
    ax[1,1].set_ylabel('z', fontsize=18)
    ax[1,1].set_title('coronal', fontsize=24)
    
    ax[1,2].imshow(y_pred[:,:,x])
    ax[1,2].set_xlabel('y', fontsize=18)
    ax[1,2].set_ylabel('z', fontsize=18)
    ax[1,2].set_title('sagittal', fontsize=24)
    
    ax[1,3].bar(hu_edges_y_pred[:-1], hist_y_pred, width=0.04, color='cornflowerblue')
    ax[1,3].set_xlabel('intensity / [HU]', fontsize=18)
    ax[1,3].set_ylabel('count', fontsize=18)
    ax[1,3].set_title('intensity distribution', fontsize=24)
    
    plt.show()    



def plot_fae_dataset(dataset: Dataset, sample: int, x: int, y: int, z: int, rescaled_age: bool, alpha: float = 0.5) -> None:
    
    image = dataset[sample][0].numpy()
    age   = dataset[sample][1]
    if rescaled_age:
        age = undo_age_rescaling(age)
    if len(dataset[sample])==3:
        sex = dataset[sample][2].numpy()
        
    print('image: {} [min = {:.1f}, max = {:.1f}]'.format(image.shape, np.min(image), np.max(image)))
    if image.shape[0] == 2:
        print('\tct:       {} [min = {:.1f}, max = {:.1f}]'.format(image[0].shape, np.min(image[0]), np.max(image[0])))
        print('\tbone_seg: {} [min = {:.1f}, max = {:.1f}]'.format(image[1].shape, np.min(image[1]), np.max(image[1])))
    print('age:   {:.1f} days or {:.1f} years'.format(age, age/365.25))
    if len(dataset[sample])==3:
        print('sex:   {} [value = {:.1f}]'.format(sex.shape, sex[0]))

    fig, ax = plt.subplots(1, 3, figsize=(24,8))
    
    ax[0].imshow(image[0,z,:,:], cmap='gist_gray')
    if image.shape[0] == 2:
        ax[0].imshow(image[1,z,:,:], cmap='Reds', alpha=alpha)
    ax[0].set_xlabel('x', fontsize=18)
    ax[0].set_ylabel('y', fontsize=18)
    ax[0].set_title('axial', fontsize=24)
    
    ax[1].imshow(image[0,:,y,:], cmap='gist_gray')
    if image.shape[0] == 2:
        ax[1].imshow(image[1,:,y,:], cmap='Reds', alpha=alpha)
    ax[1].set_xlabel('x', fontsize=18)
    ax[1].set_ylabel('z', fontsize=18)
    ax[1].set_title('coronal', fontsize=24)
    
    ax[2].imshow(image[0,:,:,x], cmap='gist_gray')
    if image.shape[0] == 2:
        ax[2].imshow(image[1,:,:,x], cmap='Reds', alpha=alpha)
    ax[2].set_xlabel('y', fontsize=18)
    ax[2].set_ylabel('z', fontsize=18)
    ax[2].set_title('sagittal', fontsize=24)
    
    plt.show()

   
    
def plot_fae_datamodule_batch(batch: list, sample: int, x: int, y: int, z: int, rescaled_age: bool, alpha: float = 0.5) -> None:
       
    image = batch[0][sample].numpy()
    age   = batch[1][sample]
    if rescaled_age:
        age = undo_age_rescaling(age)
    if len(batch)==3:
        sex = batch[2][sample].numpy()
    
    print('image: {} [min = {:.1f}, max = {:.1f}]'.format(image.shape, np.min(image), np.max(image)))
    if image.shape[0] == 2:
        print('\tct:       {} [min = {:.1f}, max = {:.1f}]'.format(image[0].shape, np.min(image[0]), np.max(image[0])))
        print('\tbone_seg: {} [min = {:.1f}, max = {:.1f}]'.format(image[1].shape, np.min(image[1]), np.max(image[1])))
    print('age:   {:.1f} days or {:.1f} years'.format(age, age/365.25))
    if len(batch)==3:
        print('sex:   {} [value = {:.1f}]'.format(sex.shape, sex[0]))
    
    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    
    ax[0].imshow(image[0,z,:,:])
    if image.shape[0] == 2:
        ax[0].imshow(image[1,z,:,:], cmap='Reds', alpha=alpha)
    ax[0].set_xlabel('x', fontsize=18)
    ax[0].set_ylabel('y', fontsize=18)
    ax[0].set_title('axial', fontsize=24)
    
    ax[1].imshow(image[0,:,y,:])
    if image.shape[0] == 2:
        ax[1].imshow(image[1,:,y,:], cmap='Reds', alpha=alpha)
    ax[1].set_xlabel('x', fontsize=18)
    ax[1].set_ylabel('z', fontsize=18)
    ax[1].set_title('coronal', fontsize=24)
    
    ax[2].imshow(image[0,:,:,x])
    if image.shape[0] == 2:
        ax[2].imshow(image[1,:,:,x], cmap='Reds', alpha=alpha)
    ax[2].set_xlabel('y', fontsize=18)
    ax[2].set_ylabel('z', fontsize=18)
    ax[2].set_title('sagittal', fontsize=24)
    
    plt.show()