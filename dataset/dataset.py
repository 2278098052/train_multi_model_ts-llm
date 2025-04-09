import sys
sys.path.append('/dataYYF/dataWX/WYL/Time-QA/')
from transformers import  PretrainedConfig, AutoTokenizer
from transformers import AutoProcessor
import torch
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import h5py
import re
from models.TimeLanguageModel import TLMConfig


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index =0
    end_index = 0
    while start_index <= len(target)-1:
        if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1,end_index+1))
                start_index=end_index+1
    return result

class TsQaDataset(Dataset):
    def __init__(self, ts_path, data_path, tokenizer, processor, config,pretrain= False,sft=False):
        super().__init__()
        self.data_path = data_path
        self.ts_path = ts_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.pretrain = pretrain
        self.sft = sft
        self.load_data()

    def load_data(self):
        if self.pretrain:
            #读取文件上一级路径
            ts_dir = os.path.dirname(self.ts_path)
            train_dir = os.path.join(ts_dir, 'train_data_Normalizd.h5')
            test_dir = os.path.join(ts_dir, 'test_data_Normalizd.h5')
            with h5py.File(train_dir, 'r') as f:
                train_data = f['seq_data'][:]
            with h5py.File(test_dir, 'r') as f:
                test_data = f['seq_data'][:]
            self.datas = np.concatenate([train_data, test_data], axis=0)

        if  not self.pretrain:
            with h5py.File(self.ts_path, 'r') as f:
            # 读取 'data_ID' 和 'seq_data' 数据集
                seq_data = f['seq_data'][:]      # 读取所有 seq_data
                id_data = f['data_ID'][:]        # 读取所有 data_ID
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.datas = []
                for line in f:
                    item = json.loads(line)
                    for i in range(0, len(item['conversations']),2):
                        query_conten = item['conversations'][i]
                        single_turn = {
                            'id': item['id'],
                            'stage':int(query_conten['stage']),
                            'form':query_conten['attribute'],
                            'question':query_conten['value'],
                            'answer':item['conversations'][i+1]['value'],
                            }   
                        # if single_turn['form'] == 'close':
                            # 只训练close的数据
                        self.datas.append(single_turn)
            for i in range(len(self.datas)):
                id = self.datas[i]['id']
                #如果id 是str:
                if isinstance(id, str):
                    id = int(id)
                    ts = seq_data[id-1]
                elif isinstance(id, list):
                    ts = []
                    # if int(id[-1]) > 75199:
                    #     # 如果这个列表中的某个值大于75199，跳过这个循环
                    #     continue
                    for index in id:
                        ts_sample = seq_data[int(index)-1]
                        ts_len = len(ts_sample)
                        ts.append(ts_sample[:int(ts_len/10), :])
                    ts = np.concatenate(ts, axis=0)
                self.datas[i]['ts'] = ts


    def __len__(self):
        return len(self.datas)
        # return 128
    def __getitem__(self, index):
        if self.pretrain:
            return {'ts_values': torch.tensor(self.datas[index],dtype=torch.float)}
        elif self.sft:
            sample = self.datas[index]
            messages = [
                {"role": "system", "content": 'You are a helpful assistant.'}, 
                {"role": "user", "content": sample['question']}
            ]

            if len(messages) >= 2 and "content" in messages[1]:
                messages[1]["content"] = '<|image_pad|>' * self.config.ts_pad_num + messages[1]["content"]

            q_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ).replace('<ts>', "")
            a_text = sample['answer'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            ts = sample['ts']

            return {
                'form': sample['form'],
                'stage': sample['stage'],
                'input_ids': input_ids,
                'labels': labels,
                'ts_values': torch.tensor(ts,dtype=torch.float)
            } 
        else:
            sample = self.datas[index]
            messages = [
                {"role": "system", "content": 'You are a helpful assistant.'}, 
                {"role": "user", "content": sample['question']}
            ]

            if len(messages) >= 2 and "content" in messages[1]:
                messages[1]["content"] = '<|image_pad|>' * self.config.ts_pad_num + messages[1]["content"]

            q_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ).replace('<ts>', "")

            a_text = sample['answer'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
    
            ts = sample['ts']
            return {
                'form': sample['form'],
                'stage': sample['stage'],
                'input_ids': q_input_ids,
                'labels': a_input_ids,
                'ts_values': torch.tensor(ts,dtype=torch.float)
            } 
     

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        ts_values = []
        forms = []
        stages = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            ts_values.append(feature['ts_values'])
            forms.append(feature['form'])
            stages.append(feature['stage'])


        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'ts_values': torch.stack(ts_values, dim=0),
            'form': forms,
            'stage': torch.tensor(stages, dtype=torch.int8)
        }

if __name__ == "__main__":
    tlmconfig = TLMConfig(llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct')
    ts_path = 'dataset/dataset_processing/data_merged_new.h5'
    qa_path = 'dataset/dataset_processing/train_sw1000.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(tlmconfig.llm_model_path)
    processor = AutoProcessor.from_pretrained(tlmconfig.llm_model_path)
    dataset = TsQaDataset(ts_path, qa_path, 
                          tokenizer, processor, tlmconfig,inference=True)
    data_collator = DataCollator(tokenizer)
    
    print("Dataset length:", len(dataset))
    print("First item in dataset:", dataset[0])
    
    features = [dataset[i] for i in range(100)]
    collated_batch = data_collator(features)
    
    print("Collated batch input_ids shape:", collated_batch['input_ids'].shape)
    print("Collated batch labels shape:", collated_batch['labels'].shape)
    print("Collated batch pixel_values shape:", collated_batch['ts_values'].shape)
