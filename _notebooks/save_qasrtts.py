import os
import sys , re
import unicodedata
import os.path as op
import pandas as pd
import numpy as np
import soundfile as sf

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk, load_dataset

num_threads = 512


train = load_from_disk('/l/users/speech_lab/QASR_TTS/QASRTTS_HF')['train']
test = load_from_disk('/l/users/speech_lab/QASR_TTS/QASRTTS_HF')['validation']
df = pd.concat([train.to_pandas(),test.to_pandas()])


def save_wav(df):
    res = ''
    for _, row in df.iterrows():
        try:
            if op.exists(row['audio_path']):
                res+=f"{row['audio_path']} exists\n"
            else:
                sf.write(row['audio_path'], row['audio'], 16000)
                res+=f"saved {row['audio_path']}\n"
        except Exception as e:
            res+="Error: {e}\n"
    return res


index = np.linspace(0, df.shape[0], num_threads, dtype=int)
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the executor for parallel audio saving
    results = [ executor.submit(save_wav, df[index[i]: index[i+1]]) for i in range(0, len(index)-1)]

    for f in as_completed(results):
        print(f.result())
        


