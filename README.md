# Exoplanet Atmosphere Prediction using a Multimodal 1D-ResNet


This repository contains my deep learning solution for predicting the atmospheric chemistry of exoplanets using noisy telescope spectra. It features a custom **1D Residual Network**, **Multimodal Data Fusion**, and **Monte Carlo Dropout** for Bayesian uncertainty estimation.

## The Physics Problem
When a planet transits in front of its host star, molecules in its atmosphere absorb specific wavelengths of starlight. By looking at the 1D transmission spectrum (transit depth across 52 wavelengths), we can theoretically deduce the planet's temperature and the abundance of 5 key molecules: $H_2O$, $CO_2$, $CH_4$, $CO$, and $NH_3$.

However, telescope noise often buries these chemical signatures. The goal of this project was to build a neural network capable of finding these molecular fingerprints while accurately quantifying its own uncertainty.

## Architecture: Multimodal 1D ResNet
Standard ML approaches flatten spatial data, which destroys the physical Signal-to-Noise Ratio (SNR). Instead, this model preserves the physics using a custom 1D ResNet.

1. **Spectral Pathway (Sequential):** The raw 52-point spectrum and its exact instrument noise profile are fed as a 2-channel 1D tensor into stacked Convolutional Residual Blocks to extract local absorption features.
2. **Stellar Pathway (Tabular):** 9 supplementary features (e.g., Star Mass, Radius, Planet Distance) are scaled and processed.
3. **Multimodal Fusion:** The extracted 1D spectral features are flattened via Adaptive Average Pooling and concatenated with the stellar features before passing through the final fully connected heads.

## Key Engineering Challenges & Solutions

### 1. Overcoming the "Lazy Network" Local Minimum
During initial training, the model quickly learned to predict Temperature (which is physically constrained by the star's heat and distance) but completely ignored the highly complex chemistry targets, resulting in a flatlined Mean Squared Error. 
* **The Fix:** I engineered a **Physics-Informed Weighted Loss Function**. By dynamically weighting the chemistry errors 100x heavier than the temperature error during backpropagation, I forced the optimizer out of the local minimum and compelled the convolutions to map the molecular absorption bands.

### 2. Estimating Uncertainty via Monte Carlo Dropout
In astrophysics, an answer without an error bar is useless. Standard neural networks only output point estimates. 
* **The Fix:** I implemented **MC Dropout** during inference. By passing the same spectrum through the network 30 times with active dropout layers, the model generated an ensemble of predictions. The variance of these predictions served as a highly calibrated, mathematically sound standard deviation ($\sigma$).

## Results & Diagnostics
The model achieved a **Gaussian Log-Likelihood (GLL) / CRPS score of 0.833**, indicating highly accurate mean predictions paired with perfectly calibrated uncertainty bounds.

### Uncertainty Calibration
*(The MC Dropout accurately predicted higher uncertainty when the physical error was higher)*
![Calibration](images/plot_calibration.png)

## Future Work / Next Steps
To push this architecture to a $>0.90$ score, future updates will include:
1. **Spectral Shape Normalization:** Dividing the 2-channel input by its own mean to isolate the relative chemical absorption shapes from the absolute transit depth.
2. **Explicit Physics Injection:** Calculating the theoretical Equilibrium Temperature ($T_{eq}$) via the Stefan-Boltzmann law and passing it directly into the tabular fusion layer.
3. **Data Augmentation:** Dynamically injecting random Gaussian noise (scaled by the telescope's actual noise profile) during training to prevent overfitting.
