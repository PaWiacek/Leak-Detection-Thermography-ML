# Analysis of Gas Leak Detection Using Thermography and Deep Learning

<img width="320" height="240" alt="thermal_split" src="https://github.com/user-attachments/assets/4c5878ba-9985-4139-a92b-cd288ca99382" />

## Overview
This project focuses on gas safety classification using thermal camera-derived data and convolutional neural networks (CNN). Leak and no-leak classes were distinguished based on experimental measurements, and the influence of reflective tube surface and thermal noise in the backround was assessed.

## Objective
This work develops a methodology for detecting gas leaks from thermal images using artificial neural networks. It identifies a suitable network architecture and evaluates the impact of thermal noise on classification performance.

## Methodology

<img width="320" height="236" alt="pneumatic-scheme-26" src="https://github.com/user-attachments/assets/a46ca354-5a07-405c-b88e-aabce69f7bc3" />

Fig. 1: *Simplified experimental scheme. The area enclosed by the dashed frame indicates the camera’s ROI (region of interest).*

To simulate gas leakages, a test bench including the main components: an air cylinder, a regulator, a valve, and a test tube was constructed. Four SUS304 steel test tubes were used in the experiments:
two with a drilled hole (leakage) and two airtight (reference samples).
Each pair consists of one tube with a reflective surface finish and one with a matte surface finish, characterized by different emissivities ϵ, measured experimentally:

| Sample   | Emmisivity value (averaged) | 
|-------------|-------------|
| Matte       | 0.17  |
| Reflective  | 0.31  |

<img width="320" height="240" alt="pipe-surface-macro-rot" src="https://github.com/user-attachments/assets/757089a4-9ad3-498d-a06c-d78bc439939e" />

Fig. 2: *Photography of pipes with leakage, top: matte, bottom: reflective surface finish*

The gas was supplied from the cylinder through the regulator under the constant pressure of 0.3 MPa, and then flows through a 0.25 inch pneumatic hose.

## Dataset
number of samples (e.g., ?? images, 2 classes)
image data augmentation
variability (angle, background, height)
the full dataset is not in the repository
## Results
## Limitations
## Future Work
## Usage
## Project Structure
