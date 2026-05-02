# Analysis of Gas Leak Detection Using Thermography and Deep Learning

<img width="320" height="240" alt="thermal_split" src="https://github.com/user-attachments/assets/4c5878ba-9985-4139-a92b-cd288ca99382" />

## Overview
This project focuses on gas safety classification using thermal camera-derived data and convolutional neural networks (CNN). Leak and no-leak classes were distinguished based on experimental measurements, and the influence of reflective tube surface and thermal noise in the backround was assessed.

## Objective
This work develops a methodology for detecting gas leaks from thermal images using artificial neural networks. It identifies a suitable network architecture and evaluates the impact of thermal noise on classification performance.

## Methodology
<img width="480" height="354" alt="pneumatic-scheme-25" src="https://github.com/user-attachments/assets/5bc0f03e-7339-4ac0-9172-5519938c3129" />
Simplified experimental scheme. The area enclosed by the dashed frame indicates the camera’s ROI (region of interest).
To simulate gas leakages, a test bench including the main components: an air cylinder, a regulator, a valve, and a test tube was constructed. Four SUS304 steel test tubes were used in the experiments:
two with a drilled hole (leakage) and two airtight (reference samples).
Each pair consists of one tube with a reflective surface finish and one with a matte surface finish, characterized by different emissivities ϵ, measured experimentally:
| Sample   | Emmisivity value (averaged) | 
|-------------|-------------|
| Reflective  | 0.31  |
| Matte       | 0.17  |

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
