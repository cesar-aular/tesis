import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

print("Testing TFT initialization...")
df = pd.DataFrame({
    'unique_id': ['1']*100 + ['2']*100,
    'ds': pd.date_range('2020-01-01', periods=100, freq='h').tolist() * 2,
    'y': range(200)
})
model = TFT(h=24, input_size=48, max_steps=2, batch_size=4, windows_batch_size=8)
nf = NeuralForecast(models=[model], freq='h')
nf.fit(df=df)
print("TFT Fit successful!")
