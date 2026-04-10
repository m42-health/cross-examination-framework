# %%
from datasets import load_from_disk
import json
import pandas as pd

# %%
model_name = "DeepSeek-V3.1"
# %%
bhc_predictions = []
with open(f"/data/cef/dischargeme/dischargeme/{model_name}/predictions/{model_name}_raw_bhc_responses_tmp.jsonl", "r") as f:
    for line in f:
        bhc_predictions.append(json.loads(line))
        
di_predictions = []
with open(f"/data/cef/dischargeme/dischargeme/{model_name}/predictions/{model_name}_raw_di_responses_tmp.jsonl", "r") as f:
    for line in f:
        di_predictions.append(json.loads(line))

# %%
data_path = "/data/cef/dischargeme/gt/discharge-me/1.3"
reference1 = pd.read_csv(os.path.join(data_path, "test_phase_1", "discharge_target.csv.gz"), keep_default_na=False)
reference2 = pd.read_csv(os.path.join(data_path, "test_phase_2", "discharge_target.csv.gz"), keep_default_na=False)
reference = pd.concat([reference1, reference2], axis=0)
# %%
