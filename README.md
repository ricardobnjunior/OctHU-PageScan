# A Fast Fully Octave Convolutional Neural Network for Document Image Segmentation

This is an implementation of OctHU-PageScan on Python 3, Keras, Tensorflow and keras-octave-conv. 
The training dataset is the CDPhotoDataset and DTDDataset. The proposed model is based on Octave Convolutions to qualify the method to situations where storage, processing, and time resources are limited, such as in mobile and robotic applications.

Informations for training:
 
 - Architecture used: OctHU-PageScan
 - Input Image Dimensions: 512x512 (gray scale - 1 channel)
 - Learning Rate: 0.0001
 - Mini-batches size: 4
 - Dimensions Output Image: 512x512 (binaeized - 1 channel)
 
 Instructions for performing the training:

 - Open file `RUN_OctHU-PageScan.py`
 - Set `train_folder`
 - Set `validation_folder`
 - Set the type of images in dataset (`type_save_image` and `type_of_images`)
 
 
### Citation

Use this BibTeX to cite this repository:

BibTeX
 
 
  
 
 