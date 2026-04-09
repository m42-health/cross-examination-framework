# %%
import requests
from datasets import load_from_disk
import sys
import os
from transformers import AutoTokenizer
import json
import uuid
# Add the parent directory to the path so we can import cef_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fire

# %%
ds = load_from_disk("/data/cef/ntrex/results_cef_new_prompt/27427907-05f7-4dd8-b6ce-2d08988b0957/results_ds")
# %%
