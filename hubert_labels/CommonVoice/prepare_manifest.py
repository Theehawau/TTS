import os
import sys , re
import unicodedata
import os.path as op
import pyarabic.araby as araby

from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from datasets import load_from_disk, load_dataset

main_path = '/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/hubert_labels/CommonVoice/'
data_root = '/l/users/speech_lab/_SpeechT5PretrainDataset/v2/data/CommonVoice/'

splits = {'train': 'train', 'test': 'test', 'dev': 'valid'}

df = load_from_disk('/l/users/speech_lab/CommonVoiceAr/CommonvVoiceHF/CommonVoice13andDelta')

map_numbers = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'}
map_numbers = dict((v, k) for k, v in map_numbers.items())
punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])
punctuations = punctuations + '÷#ݣ+=|$×⁄<>`åûݘ ڢ̇ پ'

def convert_numerals_to_digit(word):
    sentence=[]
    for w in word:
        sentence.append(map_numbers.get(w, w))
    word = ''.join(sentence)
    return word

def remove_diacritics(word):
    return araby.strip_diacritics(word)
     
def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()

def preprocess_arabic_text(text):
    text = remove_diacritics(text)
    text = convert_numerals_to_digit(text)
    text = remove_punctuation(text)
    return text

def generate_manifest(split, name):
    tsv = open(op.join(main_path, f'{name}.tsv'), 'w')
    txt = open(op.join(main_path, f'{name}.txt'), 'w')
    print(f"{data_root}", file=tsv)
    for sample in tqdm(df[split], total= len(df[split]),desc=f"Generating {name} manifest"):
        audio_path = f"CommonVoice/{sample['audio_path']}"
        spk_emb = f"speaker_embedding/{split}_{sample['speaker_id']}.npy"
        assert op.exists(op.join(data_root, audio_path)), f"{audio_path} does not exist"
        assert op.exists(op.join(data_root, spk_emb)), f"{spk_emb} does not exist"
        sr, y = wavfile.read(op.join(data_root, audio_path))
        assert sr == 16000, f"sampling rate {sr} != 16000 for {audio_path}"
        n_frames = y.shape[0]
        print(f"{audio_path}\t{n_frames}\t{spk_emb}", file=tsv)
        print(f"{preprocess_arabic_text(sample['arabic_text'])}", file=txt)

    tsv.close()
    txt.close()

def main():
    for split, name in splits.items():
        generate_manifest(split, name)


if __name__ == "__main__":
    main()
