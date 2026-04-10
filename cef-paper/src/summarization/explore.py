# %%
from datasets import load_dataset, load_from_disk

# %%
ds = load_dataset("abisee/cnn_dailymail", name="3.0.0")
# %%
# Flatten all "highlights" from all splits
ds = ds.filter(lambda x: len(x["highlights"]) > 470 and len(x["highlights"]) < 1050)

# %%
del ds["train"]
del ds["validation"]
# %%
ds = ds.map(lambda x: {"trg": x["highlights"], "src": x["article"]}, remove_columns=["highlights", "article"])
# %%
ds.save_to_disk("/data/cef/cnndm/cnndm_original")
# %%
ds = load_from_disk("/data/cef/cnndm/cnndm_original")
# %%
