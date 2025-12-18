# 2026-Big-Data-Bowl

 Description from [NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview):
 
"The downfield pass is the crown jewel of American sports. When the ball is in the air, anything can happen, like a touchdown, an interception, or a contested catch. The uncertainty and the importance of the outcome of these plays is what helps keep audiences on the edge of its seat.

The 2026 Big Data Bowl is designed to help the National Football League better understand player movement during the pass play, starting with when the ball is thrown and ending when the ball is either caught or ruled incomplete. For the offensive team, this means focusing on the targeted receiver, whose job is to move towards the ball landing location in order to complete a catch. For the defensive team, who could have several players moving towards the ball, their jobs are to both prevent the offensive player from making a catch, while also going for the ball themselves. This year's Big Data Bowl asks our fans to help track the movement of these players.

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player."

This repository implements a **multi-model framework for predicting NFL player movement trajectories** using tracking data. The project compares classical statistical models, machine learning methods, and deep learning architectures, culminating in a **Transformer-based sequence model** as the final selected approach.

---

## Project Overview

The objective is to predict a player’s future on-field trajectory (x, y positions over time) given their most recent tracking information and contextual features such as speed, acceleration, orientation, role, and distance to the ball.

Key characteristics:

* Predicts **full future trajectories**, not just final locations
* Models **Δx and Δy per frame**, then integrates forward
* Supports **frame-level, player-level, and sequence-based learning**
* Provides a **cross-validation framework** for fair model comparison

The **final selected model** is a **Transformer encoder**, tuned in `transformer_tuning_hyperparameters.R`, with animated visualizations produced in `predictions_plot.R`.

---

## Models Implemented

### 1. Generalized Additive Models (GAM)

* Thin-plate regression splines via `mgcv`
* Interpretable nonlinear baseline
* Smooth effects for speed, acceleration, distance, angle, and player attributes

### 2. XGBoost (Gradient Boosted Trees)

* Nonlinear tree-based learner
* Strong performance on tabular features
* Predicts per-frame displacement

### 3. Random Forest (Ranger)

* Ensemble of decision trees
* Captures nonlinear interactions
* Robust benchmark model

### 4. Elastic Net Regression

* Linear model with L1/L2 regularization
* Useful for feature shrinkage and baseline comparison

### 5. Physics-Based Deterministic Model

* Assumes optimal movement toward ball landing location
* Rule-based benchmark for comparison

### 6. Transformer Encoder (Final Model)

* Sequence-to-sequence Transformer architecture
* Learns temporal dependencies in player movement
* Predicts Δx and Δy at each timestep
* Achieved best cross-validated RMSE

---

## Final Model Selection

The **Transformer model** was selected as the final approach based on:

* Lowest cross-validated RMSE
* Superior long-horizon trajectory realism
* Explicit modeling of temporal structure

Hyperparameter tuning was conducted in:

```
transformer_tuning_hyperparameters.R
```

Key tuned parameters include:

* Number of attention heads
* Key/query dimension
* Feed-forward network width
* Dropout rate
* Sequence length

---

## Visualization

Trajectory animations comparing **true vs predicted player paths** are implemented in:

```
predictions_plot.R
```

These visualizations:

* Animate frame-by-frame player motion
* Overlay predictions against ground truth

---

## Repository Structure

```
├── train/
│   ├── input_2023_wXX.csv        # Tracking inputs (weekly)
│   └── output_2023_wXX.csv       # Ground truth future positions
│
├── trajectory_models.R                    # Main modeling framework
├── transformer_tuning_hyperparameters.R  # Transformer hyperparameter search
├── predictions_plot.R                     # Trajectory animation & plotting
├── README.Rmd
```

---

## Feature Engineering

Features are generated in `prepare_features()` and include:

* **Kinematics**: speed (`s`), acceleration (`a`)
* **Geometry**: distance and angle to ball
* **Directionality**: sine/cosine encodings of direction and orientation
* **Player attributes**: age, BMI, position indicators
* **Contextual roles**: target, passer, coverage
* **Temporal normalization**: frame index scaled by play length

Sequence-based models operate on fixed-length windows of these features.

---

## Cross-Validation

All models are evaluated using a unified cross-validation framework:

```r
# Example usage
set.seed(123)
cv_results <- cross_validate_models(input_data, output_data, n_folds = 5)
```

* Splits are performed at the **play level**
* Evaluation is based on full trajectory prediction
* RMSE summary statistics reported per model

---

## Evaluation Metric

Root Mean Squared Error (RMSE) over x and y coordinates:

$RMSE = \sqrt{ \frac{1}{2N} \sum ((x_{true}-x_{pred})^2 + (y_{true}-y_{pred})^2) }$

This metric captures spatial error across all predicted frames.

---

## Dependencies

Primary R packages:

* `tidyverse`, `data.table`
* `mgcv`
* `xgboost`
* `ranger`
* `glmnet`
* `caret`
* `keras3`, `reticulate`

A Python backend is required for Keras via `reticulate`.

---

## Example Usage

```r
# Example usage
set.seed(123)
cv_results <- cross_validate_models(input_data, output_data, n_folds = 5)
```

---

## Summary

This project presents a complete modeling pipeline for **NFL player trajectory prediction**, progressing from interpretable statistical models to a high-performing **Transformer-based sequence model**. The framework is modular, extensible, and suitable for advanced sports analytics research and applied modeling workflows.
