"""
cluener数据集处理:json格式转换npz
"""
import json
import os
import numpy as np
from pathlib import Path
label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30
}
import os
import json
import numpy as np
from pathlib import Path

def process(input_path):
    input_json = os.path.join(input_path, "dev.json")
    output_dir = os.path.join(input_path, "dev.npz")
    word_list = []
    label_list = []
    with open(input_json, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            text = json_line['text']
            words = list(text)
            label_entities = json_line.get('label', None)
            labels = ['O'] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            word_list.append(words)
            label_list.append(labels)
    word_list = np.array(word_list, dtype=object)
    label_list = np.array(label_list, dtype=object)
    np.savez_compressed(output_dir, words=word_list, labels=label_list)
    print("保存成功")

if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "dataset/cluener"
    process(input_path)
