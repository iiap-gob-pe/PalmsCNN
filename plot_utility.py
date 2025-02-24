
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio


def plot_training_data(features,responses,images_to_plot=3,feature_band=0,nodata_value=-9999):
  """ Tool to plot the training and response data data side by side.

  Arguments:
  features - 4d numpy array
    Array of data features, arranged as n,y,x,p, where n is the number of samples, y is the 
    data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius), 
    and p is the number of features.
  responses - 4d numpy array
    Array of of data responses, arranged as n,y,x,p, where n is the number of samples, y is the 
    data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius), 
    and p is the response dimension (always 1).
  """
  features = features.copy()
  responses = responses.copy()
  features[features == nodata_value] = np.nan
  responses[responses == nodata_value] = np.nan

  feat_nan = np.squeeze(np.isnan(features[:,:,:,0]))
  if (feature_band != 'rgb'):
    feat_min = np.nanmin(features[:,:,:,feature_band])
    feat_max = np.nanmax(features[:,:,:,feature_band])
    #features[:,:,:,feature_band] = (features[:,:,:,feature_band] - feat_min)/(feat_max-feat_min)
  else:
    for n in range(0,3):
      feat_min = np.nanmin(features[:,:,:,n])
      feat_max = np.nanmax(features[:,:,:,n])
      features = (features[:,:,:,n] - feat_min)/(feat_max-feat_min)
  features[feat_nan,:] = np.nan

  
  fig = plt.figure(figsize=(4,images_to_plot*2))
  gs1 = gridspec.GridSpec(images_to_plot, 2)
  for n in range(0,images_to_plot):
      ax = plt.subplot(gs1[n,0])
      if (feature_band == 'rgb'):
        plt.imshow(np.squeeze(features[n,:,:,:]))
      else:
        plt.imshow(features[n,:,:,feature_band],vmin=feat_min,vmax=feat_max)
      plt.xticks([])
      plt.yticks([])
      if (n == 0):
          plt.title('Feature')
  
      ax = plt.subplot(gs1[n,1])
      plt.imshow(responses[n,:,:,0])
      plt.xticks([])
      plt.yticks([])
      if (n==0):
          plt.title('Response')


def plot_saved_images(train_frame_path, train_mask_path, num_images=3, delete_after_plot=True):
    """
    Plots the first 'num_images' images from the frame and mask directories side by side and 
    optionally deletes the images from the directories after plotting.

    Args:
        train_frame_path (str): Path to the directory containing the frame images.
        train_mask_path (str): Path to the directory containing the mask images.
        num_images (int): Number of images to plot. Default is 3.
        delete_after_plot (bool): Whether to delete the images after plotting. Default is True.
    """
    
    # Get the list of image files in each directory
    frame_files = sorted(os.listdir(train_frame_path))[:num_images]
    mask_files = sorted(os.listdir(train_mask_path))[:num_images]
    
    # Create a figure to plot the images
    fig, axes = plt.subplots(num_images, 2, figsize=(5, 5))  # num_images rows, 2 columns

    # Loop through the first 'num_images' images
    for i in range(num_images):
        # Load frame image
        frame_path = os.path.join(train_frame_path, frame_files[i])
        frame_image = imageio.imread(frame_path)

        # Load mask image
        mask_path = os.path.join(train_mask_path, mask_files[i])
        mask_image = imageio.imread(mask_path)

        # Plot the frame image
        axes[i, 0].imshow(frame_image)
        axes[i, 0].set_title(f'Frame {i + 1}')
        axes[i, 0].axis('off')  # Hide axes

        # Plot the mask image
        axes[i, 1].imshow(mask_image, cmap='gray')  # Assuming masks are grayscale
        axes[i, 1].set_title(f'Mask {i + 1}')
        axes[i, 1].axis('off')  # Hide axes

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # If delete_after_plot is True, delete all files in the frames and masks directories after plotting
    if delete_after_plot:
        for file in os.listdir(train_frame_path):
            file_path = os.path.join(train_frame_path, file)
            os.remove(file_path)  # Delete frame image

        for file in os.listdir(train_mask_path):
            file_path = os.path.join(train_mask_path, file)
            os.remove(file_path)  # Delete mask image


def load_raster_band_as_uint8(file_path, band_number):
    """
    Loads a specified raster band from a GeoTIFF file and converts it to an unsigned 8-bit integer array.

    Args:
        file_path (str): Path to the GeoTIFF file.
        band_number (int): Band number to be loaded (1-based index).

    Returns:
        tuple: A tuple containing the following:
            - np.ndarray: A 2D numpy array representing the raster band, with pixel values converted to uint8.
            - list: A list containing the extent of the raster in map coordinates [xmin, xmax, ymin, ymax].
    
    Raises:
        FileNotFoundError: If the specified file cannot be opened.
        ValueError: If the specified band does not exist or no data is read from the band.
    
    Note:
        NaN values in the array are replaced with 0 to ensure consistent data handling.
    """
    # Open the raster file
    raster = gdal.Open(file_path, gdal.GA_ReadOnly)
    
    # Check if the raster is loaded successfully
    if raster is None:
        raise FileNotFoundError(f"Could not open the raster file: {file_path}. Check the file path.")

    # Get the geotransform to properly locate the raster
    geotransform = raster.GetGeoTransform()

    # Read the specified band
    band = raster.GetRasterBand(band_number)
    if band is None:
        raise ValueError(f"Error: Band {band_number} not found in {file_path}.")
    
    array = band.ReadAsArray()
    
    # Ensure the array is properly loaded before converting
    if array is None:
        raise ValueError(f"Error: No data read from band {band_number} of {file_path}.")

    # Check for NaN values and replace with 0 if any are found
    if np.issubdtype(array.dtype, np.floating) and np.any(np.isnan(array)):
        print(f"Warning: NaN values found in {file_path}, replacing with 0.")
        array = np.nan_to_num(array)

    # Convert the array to uint8
    array = array.astype(np.uint8)

    # Compute the extent of the raster (left, right, bottom, top) in map coordinates
    xmin = geotransform[0]
    xmax = xmin + geotransform[1] * raster.RasterXSize
    ymax = geotransform[3]
    ymin = ymax + geotransform[5] * raster.RasterYSize
    extent = [xmin, xmax, ymin, ymax]

    return array, extent
    
    
    
    
def plot_saved_images_with_augmentation(train_frame_path, train_mask_path, augmentation_sequence, num_images=3, delete_after_plot=True):
    """
    Plots the first 'num_images' images from the frame and mask directories side by side, applies augmentations
    to the frame images, and optionally deletes the images from the directories after plotting.

    Args:
        train_frame_path (str): Path to the directory containing the frame images.
        train_mask_path (str): Path to the directory containing the mask images.
        augmentation_sequence (iaa.Sequential): The augmentation sequence to apply.
        num_images (int): Number of images to plot. Default is 3.
        delete_after_plot (bool): Whether to delete the images after plotting. Default is True.
    """

    # Get the list of image files in each directory
    frame_files = sorted(os.listdir(train_frame_path))[:num_images]
    mask_files = sorted(os.listdir(train_mask_path))[:num_images]

    # Create a figure to plot the images
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 10))  # num_images rows, 3 columns (Original, Mask, Augmented)

    # Loop through the first 'num_images' images
    for i in range(num_images):
        # Load frame image
        frame_path = os.path.join(train_frame_path, frame_files[i])
        frame_image = imageio.imread(frame_path)

        # Load mask image
        mask_path = os.path.join(train_mask_path, mask_files[i])
        mask_image = imageio.imread(mask_path)

        # Apply augmentation to the frame image
        frame_aug = augmentation_sequence(image=frame_image)

        # Plot the original frame image
        axes[i, 0].imshow(frame_image)
        axes[i, 0].set_title(f'Original Frame {i + 1}')
        axes[i, 0].axis('off')  # Hide axes

        # Plot the mask image
        axes[i, 1].imshow(mask_image, cmap='gray')  # Assuming masks are grayscale
        axes[i, 1].set_title(f'Mask {i + 1}')
        axes[i, 1].axis('off')  # Hide axes

        # Plot the augmented frame image
        axes[i, 2].imshow(frame_aug)
        axes[i, 2].set_title(f'Augmented Frame {i + 1}')
        axes[i, 2].axis('off')  # Hide axes

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # If delete_after_plot is True, delete all files in the frames and masks directories after plotting
    if delete_after_plot:
        for file in os.listdir(train_frame_path):
            file_path = os.path.join(train_frame_path, file)
            os.remove(file_path)  # Delete frame image

        for file in os.listdir(train_mask_path):
            file_path = os.path.join(train_mask_path, file)
            os.remove(file_path)  # Delete mask image
            

def plot_images_in_pairs(directory, num_images=5):
    # Get all image files from the directory
    all_files = sorted(os.listdir(directory))

    # Initialize lists for frames and masks
    frame_files = []
    mask_files = []

    # Separate frame and mask files
    for file in all_files:
        if file.endswith('.png'): 
            if 'mask' in file:  
                mask_files.append(os.path.join(directory, file))
            else:
                frame_files.append(os.path.join(directory, file))

    # Plot the first `num_images` pairs of frames and masks
    num_images = min(num_images, len(frame_files), len(mask_files))  # Ensure we don't exceed available images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))  # num_images rows, 2 columns

    for i in range(num_images):
        # Load and plot frame image
        frame_image = imageio.imread(frame_files[i])
        axes[i, 0].imshow(frame_image)
        axes[i, 0].set_title(f'Frame {i + 1}')
        axes[i, 0].axis('off')  # Hide axes

        # Load and plot mask image
        mask_image = imageio.imread(mask_files[i])
        axes[i, 1].imshow(mask_image, cmap='gray')  # Assuming masks are grayscale
        axes[i, 1].set_title(f'Mask {i + 1}')
        axes[i, 1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()