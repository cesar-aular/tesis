---
name: advanced-lopo-validation
description: Procedural guide for Leave-One-Plant-Out (LOPO) cross-validation in solar forecasting. Use this when evaluating the zero-shot transfer learning capability of global models across multiple geographical sites.
---

# Advanced LOPO Validation

This skill provides a rigorous framework for evaluating "Cold-Start" scenarios in solar power plants.

## Workflow: Zero-Shot Transfer Learning

1. **Site Isolation**: Identify the target plant (site $k$). Ensure no data from site $k$ enters the training set of the global model.
2. **Global Training**: Train the model (e.g., TFT, NHITS) on sites $\{1, \dots, N\} \setminus \{k\}$.
3. **Inference Setup**:
   - Provide the model with the historical context (e.g., previous 96 hours) of site $k$.
   - **Crucial**: Use the model's global weights. Do not perform any fine-tuning on site $k$ to maintain "Zero-Shot" integrity.
4. **Metric Calculation**: Compute WMAPE, sMAPE, and MAE specifically for the target site $k$.

## Handling Scaling Bias
When moving from `year1` to `total` strategy, ensure that the `StandardScaler` used in the pipeline is fitted ONLY on the training sites to avoid leakage of the target site's distribution characteristics.

## Implementation Pattern (NeuralForecast)
```python
# Train on all but one
df_train = df[df['unique_id'] != target_id]
nf.fit(df=df_train)

# Predict on target using its own history for context
df_target_context = df[df['unique_id'] == target_id].iloc[:-horizon]
preds = nf.predict(df=df_target_context)
```
