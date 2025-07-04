Model Training

Dataset Preparation: Collect bright-field cell images with diverse brightness and contrast captured by JPK to ensure data diversity and representativeness, providing rich samples for subsequent training.
Labeling: Use the professional labeling tool LabelMe for pixel-level annotation of original images, defining the boundaries and category information of cell nuclei to generate label images consistent with the original dimensions.
Pyramid-UNet Construction: Build the Pyramid-UNet network structure based on PyTorch. Set parameters such as network layers, channel numbers, and convolution kernel size in net.py, and define input/output formats. data.py ensures one-to-one correspondence between original images and labels, while utils.py standardizes image sizes during training.

Model Training: Input the training set into the Pyramid-UNet model and run train.py for training. Calculate prediction results via forward propagation, compute loss against labels, and update network parameters via backpropagation. Regularly evaluate model performance using a validation set during training, adjusting hyperparameters (e.g., learning rate, batch size) based on validation results.

Image Segmentation: Run predict.py to perform semantic segmentation on cell nucleus images using the Pyramid-UNet model. After cropping and resizing, extract contours and mark nucleus boundaries, then convert to a binary image and call external scripts for fine-grained analysis. This is followed by template_matching.py, which locates probe tips via template matching, extracts nucleus centroid coordinates, defines a 250×250 pixel detection range centered on the probe, calculates the relative distance from nuclei to the probe (converted by a coefficient of -0.02e-5), and saves results to a CSV file with visual annotations.


Automated AFM Experiments

Run Run.py to monitor and process input image files in specified paths. Call the prediction script predict.py to generate output files (output_table.csv), clean temporary files, and delete original inputs after processing. It supports cyclic processing, error retry, and logs via a logger.

Run autosyn.py, an SFTP-based bidirectional file synchronization program that monitors local file system changes via Watchdog and periodically checks remote server file status to achieve automatic synchronization. The program supports operations like upload, download, and deletion, with retry mechanisms and error handling.
Run JPK.py within the JPK software to real-time acquire mechanical curves of cell nuclei within the detection range.


Additional Notes

Both synchronization and neural network codes run on the PyCharm client.

The synchronization codes autosyn.py and Run.py run simultaneously.

File synchronization uses WinSCP for data transfer via SFTP.


Reference

•https://github.com/xxxr9802/Pyramid-UNet

•https://github.com/qiaofengsheng/pytorch-UNet.git
