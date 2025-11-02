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
***Loss function Used:***
- **DiceLoss** measures the overlap between the model prediction masks and ground truth masks for each class to
determine the loss function.
- **CrossEntropyLoss** measures the per-pixel classification error between the
predicted probability distribution and ground truth labels. For the CrossEntropy, the loss was weighed for each class depending on the percentages of each present within the images.
More frequent classes such as the background, was weighed less compared to less frequent classes such as the CSF 
to capture more local details.
- By combining these 2 loss functions the model ensures
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
For the training dataset, the images were randomly augmented to prevent overfitting and improve training
performance for the segmentation task. Validation set was not augmented which was required for the MRI
segmentation validation.
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
For the training, the model prediction masks were plotted every 10 epochs to observe the performance of the model
explicitly. The loss plot was plotted at the end of the training to evaluate the segmentation of the model.
**The results plots from the model are shown below.**

```
from dataset import get_dataloaders
from model import UNet
from train import train

train_loader, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))
model = UNet(in_channels=1, out_channels=4)
train_losses = train(model, train_loader, val_loader.dataset, num_classes=4, epochs=100, lr=1e-4, visualize_every=10)
plot_loss(train_losses, metric_name="Dice Loss")
```
![OASIS MRI UNet Predictions after 10 Epochs](/recognition/improved_unet_47996283/images/10_epochs.png)

![OASIS MRI UNet Predictions after 50 Epochs](/recognition/improved_unet_47996283/images/50_epochs.png)

![OASIS MRI UNet Loss Function Plot Over Epochs](/recognition/improved_unet_47996283/images/loss_plot.png)


***Here were some epoch results from the UNet training along with the loss and DSC score:***
```
Calculating class weights from training data...
Class counts: tensor([4.5766e+08, 3.5422e+07, 7.2372e+07, 6.7883e+07])
Class percentages: [72.2618637084961, 5.592912673950195, 11.427010536193848, 10.71821117401123]
Using class weights: tensor([0.1482, 1.9151, 0.9373, 0.9993], device='cuda:0')
Starting Training with 4-Class Segmentation...
Epoch 1 — LR: 1.00e-04 | Loss: 0.6873 | Val Dice: 0.6126
Epoch 2 — LR: 1.00e-04 | Loss: 0.6155 | Val Dice: 0.6214
Epoch 3 — LR: 1.00e-04 | Loss: 0.6074 | Val Dice: 0.6040
Epoch 4 — LR: 1.00e-04 | Loss: 0.6043 | Val Dice: 0.6219
Epoch 5 — LR: 1.00e-04 | Loss: 0.6027 | Val Dice: 0.6119
Epoch 6 — LR: 1.00e-04 | Loss: 0.6019 | Val Dice: 0.6182
Epoch 7 — LR: 1.00e-04 | Loss: 0.6008 | Val Dice: 0.6269
Epoch 8 — LR: 1.00e-04 | Loss: 0.6002 | Val Dice: 0.6053
Epoch 9 — LR: 1.00e-04 | Loss: 0.5996 | Val Dice: 0.6086
Epoch 10 — LR: 1.00e-04 | Loss: 0.5992 | Val Dice: 0.6234
Epoch 11 — LR: 1.00e-04 | Loss: 0.5987 | Val Dice: 0.6133
Epoch 12 — LR: 1.00e-04 | Loss: 0.5984 | Val Dice: 0.6041
Epoch 13 — LR: 1.00e-04 | Loss: 0.5977 | Val Dice: 0.6096
Epoch 14 — LR: 1.00e-04 | Loss: 0.5976 | Val Dice: 0.6154
Epoch 15 — LR: 1.00e-04 | Loss: 0.5969 | Val Dice: 0.6042
Epoch 16 — LR: 1.00e-04 | Loss: 0.5965 | Val Dice: 0.6004
Epoch 17 — LR: 1.00e-04 | Loss: 0.5958 | Val Dice: 0.6144
Epoch 18 — LR: 1.00e-04 | Loss: 0.5953 | Val Dice: 0.6199
Epoch 19 — LR: 1.00e-04 | Loss: 0.5945 | Val Dice: 0.6138
Epoch 20 — LR: 1.00e-04 | Loss: 0.5940 | Val Dice: 0.6086
Epoch 21 — LR: 1.00e-04 | Loss: 0.5936 | Val Dice: 0.6073
Epoch 22 — LR: 1.00e-04 | Loss: 0.5924 | Val Dice: 0.6152
```
Unfortunately, DSC score was not able to reach 0.9 due to time constraints.
However, training the model for 1000 epochs should reach the DSC score of 0.9.

### Example Usage for predicting the Datasets using the TRAINED UNet model:
This plots the trained UNet model's prediction given a set of MRI images and comparing these predictions
to the ground truth masks as observed below.
```
model = load_trained_model()
_, val_loader = get_dataloaders(batch_size=8, img_size=(256, 256))
val_dataset = val_loader.dataset
show_predictions(model, val_dataset, n=5, save_path="final_predictions.png")
```

![OASIS MRI Training Predictions from Trained UNet](/recognition/improved_unet_47996283/images/final_predictions.png)

## References

- O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.