---
name: transformer-interpretability
description: Workflows for extracting and visualizing attention weights from Temporal Fusion Transformers (TFT). Use this to provide eXplainable AI (XAI) insights for the thesis.
---

# Transformer Interpretability (XAI)

This skill focuses on the "Attention" mechanism of the TFT model.

## Variable Selection Weights
TFT uses a Variable Selection Component. Extracting these weights allows us to rank features:
1. **Encoder Importance**: Which historical variables (past GHI, past Power) the model prioritizes.
2. **Decoder Importance**: Which future exog (forecasted GHI) dominates the prediction.

## Temporal Attention
Identify which past time-steps (e.g., $t-24$, $t-168$) the model "looks at" when making a forecast.
- **Pattern Recognition**: High attention at $t-24$ confirms the model has learned the daily cycle.
- **Anomaly Detection**: Unusual attention shifts during cloudy days indicate the model is adapting to weather volatility.

## Visualizing Weight Matrices
Use heatmaps to represent the attention weights across the look-back window. 
- **Mandate**: Normalize weights per head to ensure visual comparability.
