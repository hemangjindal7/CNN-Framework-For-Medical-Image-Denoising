This repository presents a deep learning-based approach to image denoising, aiming to restore clarity by reducing noise in grayscale images. The model employs a U-Net architecture with perceptual loss, offering effective noise reduction while preserving key details.

Testing New Images
How to Use
References
Project Overview
Image denoising aims to remove unwanted noise from images, enhancing their visual quality and usability. This project uses a U-Net-based autoencoder model, which learns the mapping between noisy and clean images, leveraging perceptual loss to improve output quality.

Dataset
Dataset Structure: The dataset comprises paired noisy and clean grayscale images.
Noise Addition: Salt-and-pepper noise is added to clean images for training, simulating real-world noise.
Image Preprocessing: Each image is resized to 128x128 pixels, normalized to a [0, 1] range, and reshaped to grayscale format.
Methodology
Data Preprocessing
Image Loading: Images are loaded from a specified directory, resized, normalized, and converted to grayscale.
Noise Injection: Salt-and-pepper noise is applied to a subset of pixels in each image, generating training data for the denoising task.
Data Splitting: The dataset is split into training, validation, and test sets with an 80-10-10 ratio.
Model Architecture
A U-Net autoencoder is used for denoising, structured as follows:

Encoder: A series of convolutional and pooling layers extract features from the noisy images.
Bottleneck: Deep feature representations are learned, acting as a compression stage.
Decoder with Skip Connections: Deconvolution layers restore spatial resolution, incorporating skip connections to retain spatial details.
Training Process
Loss Function: Perceptual loss, calculated using a pre-trained VGG16 network, is applied to encourage high-level feature similarity between the denoised and clean images.
Optimizer: The Adam optimizer is used for efficient training and convergence.
Callbacks: Early stopping and model checkpointing are implemented to monitor validation loss and save the best-performing model.
Evaluation
Quantitative Metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) are used to evaluate the model's performance, comparing the denoised images against ground truth.
Qualitative Assessment: A visual comparison between noisy, denoised, and sharpened images showcases the effectiveness of the model.
Results
Performance: The model achieves high PSNR and SSIM scores, indicating successful noise reduction.
Visual Output: Sample results show significant improvements in image clarity and detail retention after denoising.
Testing New Images
To test new noisy images with the trained model:

Load a grayscale image.
Resize and normalize the image to the model's input specifications.
Use the model to predict the denoised version, optionally applying a sharpening filter to enhance detail.
Display the results for a comparative view.











