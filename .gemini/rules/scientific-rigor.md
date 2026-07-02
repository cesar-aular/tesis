# Scientific Rigor Mandates (From Plan.md & Anteproyecto)

1.  **Epoch-Based Training:** Models must perform a statistically significant number of passes over the *entire* dataset. Training steps must be calculated dynamically: $Steps = (TotalRows / (BatchSize \times StepSize)) \times Epochs$.
2.  **Probabilistic Forecasting:** The solution is not a point-forecast. It must use **Quantile Loss (P10, P50, P90)** to provide uncertainty bands (Fan Charts), critical for grid stability as well as the specific number.
3.  **Zero-Shot Integrity:** In Leave-One-Plant-Out (LOPO) mode, the target plant must remain strictly unseen. Any data leakage (using target plant stats for global scaling) is a critical failure.
4.  **Physical Consistency:** Forecasts must respect solar physics (e.g., zero generation during night hours, GHI/Power correlation).
5.  **Data Leakage Prevention:** Always verify mathematically that the target variable (`y`) does not contaminate the training dataset.
