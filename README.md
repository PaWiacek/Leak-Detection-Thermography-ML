# Analysis of Gas Leak Detection Using Thermography and Deep Learning

<img width="320" height="240" alt="thermal_split" src="https://github.com/user-attachments/assets/4c5878ba-9985-4139-a92b-cd288ca99382" />

## Overview
This project focuses on gas safety classification using thermal camera-derived data and convolutional neural networks (CNN). Leak and no-leak classes were distinguished based on experimental measurements, and the influence of reflective tube surface and thermal noise in the backround was assessed.

## Key Results

- Self-collected experimental dataset
- Accuracy: 65% on test subset and 69% on validation subset
- Robustness to background noise evaluated
- Best model selected via Bayesian optimization

## Objective
Although several gas leak detection technologies are already available, including optical gas imaging (OGI) cameras, catalytic combustible gas sensors (CGS), ultrasonic sensors, and fiber-optic sensors, they often face limitations in terms of detection speed, accuracy, or robustness to environmental conditions. Therefore, there is still a need to develop fast and reliable detection methods.

This work develops a methodology for detecting gas leaks from thermal images using artificial neural networks. It identifies a suitable network architecture and evaluates the impact of thermal noise on classification performance.

## Methodology
<img width="320" height="236" alt="pneumatic-scheme-26" src="https://github.com/user-attachments/assets/a46ca354-5a07-405c-b88e-aabce69f7bc3" />

Fig. 1: *Simplified experimental scheme. The area enclosed by the dashed frame indicates the camera’s ROI (region of interest).*

<img width="320" height="213" alt="experimental-device" src="https://github.com/user-attachments/assets/61573a62-f7a7-4094-9f09-46d35d4a5583" />

Fig. 2: *Experimental device with test tube and 3D-printed elements. Both cylinder and regulator are not visible.*


To simulate gas leakages, a test bench including the main components: an air cylinder, a regulator, a valve, and a test tube was constructed. Four SUS304 steel test tubes were used in the experiments:
two with a drilled hole (leakage) and two airtight (reference samples).
Each pair consisted of one tube with a reflective surface finish and one with a matte surface finish, characterized by different emissivities ϵ, measured experimentally:

| Sample   | Emmisivity value (averaged) | 
|-------------|-------------|
| Matte       | 0.17  |
| Reflective  | 0.31  |

<img width="320" height="240" alt="pipe-surface-macro-rot" src="https://github.com/user-attachments/assets/757089a4-9ad3-498d-a06c-d78bc439939e" />

Fig. 3: *Photography of pipes with leakage, top: matte, bottom: reflective surface finish*

The gas was supplied from the cylinder through the regulator under the constant pressure of 0.3 MPa, and then flows through a 0.25 inch pneumatic hose. Each test tube was connected using Swagelock fittings to the rest of installation.

The primary measurement device of the setup was a FLIR T630sc thermal camera, with essential specification parameters and its values listed below:

| Parameter                  | Value                         |
|----------------------------|-------------------------------|
| Thermal resolution         | 640 × 480 px                  |
| Thermal sensitivity (NETD) | < 30 mK                       |
| Accuracy                   | ±2 °C or 2% at 25 °C          |
| Lens focal length          | 24.6 mm                       |

## Neural network architecture
<img width="711" height="200" alt="net_layers" src="https://github.com/user-attachments/assets/4c42c51f-6ffe-48c4-a023-41561583bc65" />

Fig. 4: *Schematic representation of network layers.*

The network takes RGB thermal images of size 640×480×3 as input. It consists of two convolutional blocks, each including a convolutional layer (with optimized filter size and number of filters, using same padding), batch normalization, a ReLU activation, and max pooling for spatial downsampling. The pooling size is fixed at 2, while the stride is optimized.

A dropout layer follows, randomly deactivating a fraction of activations based on an optimized dropout rate. The final layers include a fully connected layer with two outputs, a softmax layer producing class probabilities, and a classification layer assigning the input to either the leak or no-leak class.

## Dataset
To gather a diverse dataset for the CNN, thermograms were acquired for different camera heights z, sample orientations φ and camera viewing angles θ, ensuring that individual images differed in their geometric configuration and reducing the risk of data leakage during subsequent machine learning (ML) stage.

<img width="320" height="227" alt="z_determin" src="https://github.com/user-attachments/assets/637b5295-ec59-476d-a277-a9761c75c6b6" />

Fig. 5: *Tripod height z determination; side view.*

<img width="320" height="117" alt="phi_determin-crop-1" src="https://github.com/user-attachments/assets/1f2bcb8d-199a-408f-b6e6-6121acbcfdae" />

Fig. 6: *φ angle (tube orientation) determination; front view.*

<img width="320" height="227" alt="theta_determin" src="https://github.com/user-attachments/assets/5ead940d-6fa5-472d-830c-0c0a818fef5e" />

Fig. 7: *θ angle determination; top view.*

- Number of samples: 360 unique images split across 3 subsets:
  - train (216 images)
  - test (72 images)
  - validation(72 images)
- variable parameters:
  - tripod angle θ
  - sample (tube) angle φ
  - camera height z
- controlled factors evaluated for their impact on accuracy:
  - surface emmisivity (ϵ 0.17 vs ϵ 0.31)
  - background type (laboratory vs screen)
- Training image data augmentation: from original 360 samples to 412 in total
- The full dataset is not included in the repository. Further information is available in the README file in the dataset_samples directory.

## Results

<img width="320" height="225" alt="acc-comp-largeNet" src="https://github.com/user-attachments/assets/38d99f57-b8be-43ae-83d9-9788ca9935f9" />

Fig. 8: *Accuracy comparison for each subset*

<img width="320" height="258" alt="confMat-largeNet" src="https://github.com/user-attachments/assets/c372689c-7e81-4541-86e2-c4fbc2bc5165" />

Fig. 9: *Confusion matrix*

<img width="320" height="230" alt="mismatched_images" src="https://github.com/user-attachments/assets/e7d104cc-0a56-4870-b312-582f6ec974f8" />

Fig. 10: *Mismatched images by sample type*


<!-- ## Limitations -->

<!-- ## Future Work -->
## Usage
### HPC (Slurm, Bayesian Optimization)
Experiments can be run on an HPC cluster using Slurm scripts provided in the hpc/ directory.

Submit the job:

```bash
sbatch hpc/run_experiment_complex_bn_fc.sh
```

The script:
- launches `run_main_optimization.m` with predefined settings (such as time and allocated resources)
- performs Bayesian optimization of hyperparameters (e.g., filter size, number of filters, dropout)
- calls the objective function `cnnObjectiveFcn.m` for training and evaluation
- loads the network architecture from `createCNN.m`
- saves the model/results if improved performance is achieved

<!-- The objective function (`cnnObjectiveFcn`) trains the network defined in `createCNN` and returns validation metrics used by the optimizer. -->

### Local (MATLAB)
1. Open MATLAB in the `src/` directory
2. Run:
   ```matlab
   CNN_large.m
   ```
3. Evaluates the performance of the pre-trained network without retraining

<!-- ## Project Structure -->
