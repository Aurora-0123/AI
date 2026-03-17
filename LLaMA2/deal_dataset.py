import os
import json
from tqdm import tqdm

pretrain_data = 'D:\LLM\LLaMA2\mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = 'seq_monkey_datawhale.jsonl'

sft_data = 'D:/LLM/LLaMA2/BelleGroup/train_3.5M_CN.json'
output_sft_data = 'BellGroup_sft.jsonl'

def split_text(text,chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data,'a',encoding='utf-8') as pretrain:
    with open(pretrain_data,'r',encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data,desc=f"Processing lines in {pretrain_data}",leave=False):
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text':chunk},ensure_ascii=False)+'\n')

def convert_message(data):
    message = [
        {"role": "system","content": "你是一个AI助手"}
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({"role": "human","content": item['value']})
        elif item['from'] == 'assistant':
            message.append({"role": "assistant","content": item['value']})
    return message

with open(output_sft_data,'a',encoding='utf-8') as sft:
    with open(sft_data,'r',encoding='utf-8') as f:
        data = f.readlines()
        for item in tqdm(data,desc="Processing",unit="line"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message,ensure_ascii=False)+'\n')