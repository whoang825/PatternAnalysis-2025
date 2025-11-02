# Improved UNET for Segmenting the 2D OASIS Dataset

### Chosen Task

Segment the 2D OASIS brain data set with an Improved UNet with all labels having a
minimum Dice similarity coefficient of 0.9 on the test set.

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

![UNet Architecture](/recognition/improved_unet_47996283/images/unet_architecture.png)

Each input MRI image was resized and normalised to 256x256 and the segmentation masks were encoded
into discrete classes from 0 to 3. During the training, the model uses a **combination** of CrossEntropyLoss and
DiceLoss for the loss function and minimizes this loss using gradient descent and backpropagation via the **Adam**
optimizer. The UNet model outputs a mask for each class for the MRI slices (what it predicts) after the upsampling and 
downsampling. 
**DiceLoss** measures the overlap between the model prediction masks and ground truth masks for each class to
determine the loss function. **CrossEntropyLoss** measures the per-pixel classification error between the
predicted probability distribution and ground truth labels. By combining these 2 loss functions the model ensures
pixel level learning along with the mask region overlapping which is essential for the segmentation.


## Dependencies

| Package | Version |
|----------|----------|
| Python | 3.10+ |
| PyTorch | 2.1.0 |
| torchvision | 0.16.0 |
| numpy | 1.26 |
| matplotlib | 3.8 |

## Example Usage and Results

The OASIS MRI Image Datasets were loaded into the training and validation set loaders while on
the Rangpur cluster and training, validation and testing was also done on the Rangpur.
(Data was NOT retrieved from a project folder. Data was retrieved from the
Rangpur Path: /home/groups/comp3710/OASIS)

### Example Usage for loading the Datasets:
```
tl, vl = get_dataloaders(batch_size=4)
train_dataset = tl.dataset
val_dataset = vl.dataset
show_examples(train_dataset, title="MRI Training Dataset + Binary Masks", n=3, save_path="train_examples.png")
show_examples(val_dataset, title="MRI Validation Dataset + Binary Masks", n=3, save_path="validation_examples.png")
```
![OASIS MRI Training Dataset + Masks](/recognition/improved_unet_47996283/images/train_examples.png)
![OASIS MRI Validation Dataset + Masks](/recognition/improved_unet_47996283/images/validation_examples.png)

### Example Usage for the training:
```
from dataset import get_dataloaders
from model import UNet
from train import train

train_loader, val_loader = get_dataloaders(batch_size=8)
model = UNet(in_channels=1, out_channels=4)
train(model, train_loader, val_loader.dataset, epochs=20, lr=1e-3)
```

## References

- O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.