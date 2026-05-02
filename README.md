# Analysis of Gas Leak Detection Using Thermography and Deep Learning

<img width="320" height="240" alt="thermal_split" src="https://github.com/user-attachments/assets/4c5878ba-9985-4139-a92b-cd288ca99382" />

## Overview
This project focuses on gas safety classification using thermal camera-derived data and convolutional neural networks (CNN). Leak and no-leak classes were distinguished based on experimental measurements, and the influence of reflective tube surface and thermal noise in the backround was assessed.

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

## Dataset
To gather a diverse dataset for the CNN, thermograms were acquired for different camera heights z, sample orientations φ and camera viewing angles θ, ensuring that individual images differed in their geometric configuration and reducing the risk of data leakage during subsequent machine learning (ML) stage.

- Number of samples: 360 unique images split across 3 subsets: train (216 images), test (72 images), validation(72 images), and further into 2 classes (leak/no leak)

train/ (216 images total)
|-- leak/
`-- noleak/
val/ (72 images total)
|-- leak/
`-- noleak/
test/ (72 images total)
|-- leak/
`-- noleak/

- variability (angle, background, height)
- Training image data augmentation: from original 360 samples to 412 in total


the full dataset is not in the repository
## Results
## Limitations
## Future Work
## Usage
## Project Structure
