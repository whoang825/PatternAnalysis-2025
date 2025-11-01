# Improved UNET for Segmenting the 2D OASIS Dataset

### Chosen Task

Segment the 2D OASIS brain data set with an Improved UNet with all labels having a
minimum Dice similarity coefficient of 0.9 on the test set. [Easy Difficulty]

## Task Description

This task was conducted by implementing a **UNet Convolutional Neural Network** for
automated brain MRI segmentation, a critical aspect in medical image analysis
for detecting brain tissue and abnormalities. 
Accuracy and reliability of the segmentation of the MRI images is significant for
tumor location and neurological studies.
The UNet model learns to segment MRI slices from OASIS into 4 distinct classes:
- Gray matter
- White matter
- Cerebrospinal fluid
- Background

This was done by extracting the key features and contextual information using convolution
by downsampling and upsampling the images with the UNet encoder and decoder architecture.

## UNet CNN Model Explanation

The UNet model consists of an encoder, decoder and the skip connections for the image segmentation
task:
- The encoder downsamples the images, capturing the spatial context from the convolution and max pooling 
layers. These would include the more global features such as the brain shape, brain tissue structure, etc.
- The decoder upsamples the images, capturing more of the spatial and finer details by the transposed
convolution layers and the skip connections. These would include more finer and local details of the MRI
image such as abnormalities in tissue, etc.
- The skip connections link the encoder and decoder of the same resolution to improve localization accuracy.
