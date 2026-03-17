import argparse
import random
import json
import os
from distutils.command.config import config
from tqdm import tqdm

from pyexpat.errors import messages
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import(
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

random.seed(42)

def read_text_from_jsonl(file_path:str, max_bytes: int = None) -> Generator[str, None, None]:
    file_size = os.path.getsize(file_path)
    total_size = max_bytes if max_bytes is not None and max_bytes < file_size else file_size
    processed_bytes = 0
    with open(file_path, "r", encoding="utf-8") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing Data") as pbar:
            for line_num, line in enumerate(f,1):
                line_bytes = (len(line.encode('utf-8')))
                processed_bytes += line_bytes

                # 更新进度条
                pbar.update(line_bytes)

                try:
                    data = json.loads(line)
                    if 'text' not in data:
                        raise KeyError(f"Missing 'text' field in line {line_num}")
                    yield data['text']
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in line {line_num}")
                    continue
                except KeyError as e:
                    print(e)
                    continue
                if max_bytes is not None and processed_bytes >= max_bytes:
                    print(f"\nReached max_bytes limit ({max_bytes / 1024 ** 3:.2f} GB). Stopping.")
                    break

def create_tokenizer_config(save_dir:str) -> None:
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_space": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template":(
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{message['content']}}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{message['content']}}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{message['content']}}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generator_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<s>","</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path:str, save_dir:str,vocab_size:int = 8192) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 配置特殊tokens
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
        min_frequency = 2,
        show_progress = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )

    # === 设置 5GB 限制 ===
    target_gb = 0.5
    max_bytes = target_gb * 1024 * 1024 * 1024  # 转换成字节

    print(f"Training tokenizer with first {target_gb} GB of data from {data_path}")

    # 传入 max_bytes 参数
    texts = read_text_from_jsonl(data_path, max_bytes=max_bytes)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error", e)
        raise

    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path:str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer:{e}")
        return

    # 测试基本信息
    print("\n=== Tokenizer基本信息 ====")
    print(f"Vocab size:{len(tokenizer)}")
    print(f"Special tokens:{tokenizer.all_special_tokens}")
    print(f"Special token IDs:{tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role":"system","content":"你是一个AI助手。"},
        {"role":"user","content":"How are you?"},
        {"role":"assistant","content":"I'm fine,thank you.And you?"},
        {"role":"user","content":"I'm good too."},
        {"role":"assistant","content":"That's great to hear!"}
    ]

    print("\n===聊天模板测试===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n===编码解码测试===")
    encoded = tokenizer(prompt, truncation=True, max_length=512)
    decoded = tokenizer.decode(encoded["input_ids"],skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n===特殊token处理===")
    test_texts = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_texts).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original:{test_texts}")
    print(f"Decoded:{decoded}")
    print("Special tokens preserved:", test_texts == decoded)

def main():
    data_path = "D:\LLM\LLaMA2\mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = "tokenizer_k"

    # 训练tokenizer
    train_tokenizer(
        data_path,
        save_dir,
        vocab_size = 6144
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)

if __name__ == "__main__":
    main()