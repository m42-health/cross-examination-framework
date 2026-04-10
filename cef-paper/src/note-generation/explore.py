# %%
from datasets import load_from_disk

# %%
ds = load_from_disk("/data/evaluation/medic/summarization_evaluation/datasets/aci-bench-final/")
# %%
ds = ds.map(lambda x: {"id": x["encounter_id"], "src": x["dialogue"], "trg": x["note"]})
ds = ds.select_columns(["id", "src", "trg"])
del ds["train"]
# %%
ds.save_to_disk("/data/cef/aci/aci_original")
# %%
ds = load_from_disk("/data/cef/aci/aci_original")
# %%
