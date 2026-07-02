import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]

for mod in models:
    test_path = Path(f"src/ml/{mod}/test.py")
    if test_path.exists():
        with open(test_path, "r") as f:
            content = f.read()
            
        new_content = content.replace("preds_day1 = nf.predict(df=hist_df, futr_df=futr_df_day1_input)", "static_df = test_df[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()\n    preds_day1 = nf.predict(df=hist_df, futr_df=futr_df_day1_input, static_df=static_df)")
        new_content = new_content.replace("preds = nf.predict(df=current_hist, futr_df=futr_day_input).reset_index()", "preds = nf.predict(df=current_hist, futr_df=futr_day_input, static_df=static_df).reset_index()")
        
        with open(test_path, "w") as f:
            f.write(new_content)
            
print("Added static_df to predict calls.")
