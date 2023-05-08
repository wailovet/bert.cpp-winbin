import sys
import struct
import json
import torch
import numpy as np

from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "paraphrase-multilingual-MiniLM-L12-v2",
)

tokenizer.save_vocabulary("test_dir")
tokenizer.save_pretrained("test_dir")

print("test:", tokenizer.encode('你好 你叫什么名字')) 