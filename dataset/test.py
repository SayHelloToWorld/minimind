from datasets import load_dataset
import traceback
try:
    ds = load_dataset('json', data_files='dataset/sft_t2t.jsonl', split='train')
    print('Loaded', len(ds), 'samples')
except Exception as e:
    traceback.print_exc()