#!/usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
from transformers import Trainer, TrainingArguments, AdamW
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
import os
import torch
from models.TimeLanguageModel import TLM, TLMConfig
from dataset.dataset import DataCollator
from typing import Dict, List, Any, NamedTuple, Optional, Tuple, Union
from datasets import load_metric
import numpy as np
from utils.metrics import open_question_metrics,closed_question_metrics,compute_rul
import warnings
from tqdm import tqdm
import pickle
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
class OutputWrapper:
    def __init__(self, original_output):
        self.original_output = original_output

    def __getattr__(self, name):
        # 如果属性不存在于自身，则尝试从原始对象中获取
        return getattr(self.original_output, name)
    
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    pred_extra: Optional[Dict[str, Any]] = None

class Exp_Instruct(Trainer):
    def __init__(self, args, train_dataset,tlm_config =None, eval_dataset=None):
        # Build the model
        self.tlmconfig = tlm_config
        model = self._build_model(args)
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_strategy='no',
            eval_steps=args.eval_steps,
            save_total_limit=2,
            num_train_epochs=args.num_train_epochs,
            report_to=args.report_to,  # Example: Integrate TensorBoard
            prediction_loss_only=False)

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollator(tokenizer=train_dataset.tokenizer),
            eval_dataset=eval_dataset,
            # compute_metrics=self._compute_metrics if eval_dataset else None,
        )
        self.compute_metrics  = self.custom_compute_metrics if eval_dataset else None
        self.special_id = train_dataset.processor.all_special_ids  
        self.tokenizer = train_dataset.processor
        self.padding_idx = self.tokenizer.pad_token_id
        # 常用标点符号列表
        common_punctuations = [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "\"", "'"]
        punctuation_ids = self.tokenizer.convert_tokens_to_ids(common_punctuations)
        # 将标点符号 ID 合并到特殊标记 ID 列表中
        self.special_id.extend(punctuation_ids)
        self.tlmargs = args
        self.loss_fn = nn.CrossEntropyLoss()
        # self.args.remove_unused_columns = True  # 添加这一行
    def load_model(self, checkpoint_path):
        self.model = TLM.from_pretrained(checkpoint_path, config=self.tlmconfig, ts_config=self.tlmargs).cuda()

    def _build_model(self, args):
        """Load the model dynamically based on the configuration."""

        # self.tlmconfig = TLMConfig(llm_model_path = args.llm_model_path)
        model = TLM(self.tlmconfig,args).cuda()
        print(model)
        print(f'模型训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        return model
    
    def concat_np_array(self, array_list,num_samples):
        """
        对传入的列表进行 Concat 操作。
        
        Args:
            array_list (List[List[int]]): 每个子列表为需要 Padding 的序列。
            num_samples (int): 样本数量。
        Returns:
            np.ndarray: Padding 后的二维数组。
        """
        # 获取最大长度
        max_length = max(arr.shape[-1] for arr in array_list)
        
        # 初始化 Padding 后的数组，填充为 padding_idx
        padded_array = np.full((num_samples, max_length), self.padding_idx, dtype=np.int32)
        
        # 填充每个序列
        for i, arr in enumerate(array_list):
            padded_array[:arr.shape[0], :arr.shape[1]] = arr
        concat_array = np.stack(padded_array, axis=0)
        return concat_array    
    def plot_tensor_lines(self,tensor):
        """
        将给定的tensor从GPU转到CPU，转换为DataFrame，并绘制多条线图
        
        参数:
            tensor: 输入的PyTorch tensor，形状应为(time_points, lines)
        """
        # 确保输入是tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # 获取tensor的形状 (time_points, lines)
        time_points, lines = tensor.shape
        
        # 将tensor转到CPU并转换为numpy数组
        tensor_cpu = tensor.cpu().numpy()
        
        # 转换为DataFrame
        df = pd.DataFrame(tensor_cpu)
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制每条线
        for i in range(lines):
            plt.plot(df[i], label=f'Line {i+1}')
        
        # 添加图例和标签
        plt.title(f'{lines} Lines over {time_points} Time Points')
        plt.xlabel('Time Points')
        plt.ylabel('Values')
        
        # 如果线条太多，可以简化图例显示
        if lines > 20:
            plt.legend(ncol=3, fontsize='small')
        else:
            plt.legend()
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        plt.savefig('/dataYYF/dataWX/SJ/Time-QA/save/tensor_plot.png', dpi=600)
    def sample_top_p(self,probs, p):
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    def prediction_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        all_predictions = []
        all_labels = []
        all_losses = []

        model = self._wrap_model(self.model, training=False)
        model.eval()
        sample_num = len(dataloader.dataset)
        forms = []
        stages = []
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader, desc=description)):
                s = inputs['input_ids'].shape[1]
                # inputs['ts_values']=torch.randn((8,600,33),dtype=torch.float32)
                while inputs['input_ids'].shape[1]<s+100-1:
                    inputs = self._prepare_inputs(inputs)
                    outputs = model(**inputs).logits
                    logits = outputs[:, -1, :] 

                    for token in set(inputs["input_ids"].tolist()[0]):  
                        logits[:, token] /= 1.0

                    logits = logits / 0.8   #temperture=0.8 
                    v, _ = torch.topk(logits, min(5, logits.size(-1)))  #top-k=5
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                    probs = nn.functional.softmax(logits, dim=-1)  
                    idx_next = torch.multinomial(probs, num_samples=1, generator=None) 
                    is_151643 = (inputs["input_ids"] == 151643)  
                    all_samples_have_151643 = is_151643.any(dim=1).all()  
                    
                    if all_samples_have_151643:
                        print("所有样本均已生成151643，终止生成")
                        break

                    inputs["input_ids"] = torch.cat((inputs['input_ids'], idx_next), dim=1)
                print(self.tokenizer.decode(inputs["input_ids"][0]))

                prediction = inputs['input_ids'].cpu().numpy()
                all_predictions.extend(prediction)
                all_labels.extend(inputs["labels"].cpu().numpy())

                forms.extend(inputs['form'])
                stages.extend(inputs['stage'].tolist())

        filtered_preds, filtered_labels = [], []
        for pred, label in zip(all_predictions, all_labels):

            matches = np.where(pred==151643)
            if len(matches[0])>0:
                first_match_idx = matches[0][0]
                truncated_sample = pred[:first_match_idx]
            else:
                truncated_sample = pred
            # 找到 label 中所有非特殊 token 的位置
            valid_positions_for_label = [i for i, token in enumerate(label) if token not in self.special_id]
            valid_positions_for_predict = [i for i,token in enumerate(truncated_sample) if token not in self.special_id]
            # 从 predictions 和 labels 中提取对应的 token
            filtered_preds.append([truncated_sample[i] for i in valid_positions_for_predict])
            filtered_labels.append([label[i] for i in valid_positions_for_label])
        
        #将所有的id转化为文本
        str_predictions = self.tokenizer.batch_decode(filtered_preds)
        str_labels = self.tokenizer.batch_decode(filtered_labels)

        # forms = [d['form'] for d in dataloader.dataset][:sample_num]
        # stages = [d['stage'] for d in dataloader.dataset][:sample_num]
        pred_extra = {'forms': forms, 'stages': stages}
        avg_loss = np.mean(all_losses) if all_losses else None

        with open('eval_output_7B.pkl','wb') as f:
            pickle.dump(EvalLoopOutput(predictions=all_predictions, label_ids=all_labels,
                               metrics=avg_loss, num_samples=sample_num
                               ,pred_extra=pred_extra),f)


        return EvalLoopOutput(predictions=str_predictions, label_ids=str_labels,
                               metrics=avg_loss, num_samples=sample_num
                               ,pred_extra=pred_extra)
    
    def generate(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        all_predictions = []
        all_labels = []
        all_losses = []

        model = self._wrap_model(self.model, training=False)
        model.eval()
        sample_num = len(dataloader.dataset)
        forms = []
        stages = []
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader, desc=description)):
           
                bsz = inputs['input_ids'].shape[0]
                mask = (inputs['input_ids'] == 151643)
                first_pad_indices = mask.int().argmax(dim=1)
                no_mask = ~mask.any(dim=1)
                first_pad_indices[no_mask] = inputs['input_ids'].size(1)
                min_prompt_len = first_pad_indices.min()-1
                max_prompt_len = first_pad_indices.max()

                total_len = max_prompt_len + 50-1
                pad_id = 151643#<endoftext>
                eos_reached = torch.tensor([False] * bsz,device=inputs['input_ids'].device)
                

                temperature = 0.8
                top_p = 0.6
                origin_input = torch.full(
                    (inputs['input_ids'].shape[0], total_len+1),
                    fill_value=pad_id,
                    dtype=inputs['input_ids'].dtype,
                    device=inputs['input_ids'].device
                )
                origin_input[:, :inputs['input_ids'].shape[1]] = inputs['input_ids']
                input_text_mask = origin_input != pad_id
                for cur_pos in range(min_prompt_len, total_len):
                    inputs['input_ids']=origin_input[:, 0:cur_pos+1]
                    logits = model.forward(**inputs,mode='inference')
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = self.sample_top_p(probs, top_p)#(8,1)


                    next_token = next_token.reshape(-1)
                    # only replace token if prompt has already been generated

                    next_token = torch.where(
                        input_text_mask[:, cur_pos+1], origin_input[:, cur_pos], next_token
                    )
                    origin_input[:, cur_pos+1] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos+1]) & (
                        next_token == torch.tensor(151645,device=next_token.device) #<|im_end|>
                    )
                    if all(eos_reached):
                        inputs['input_ids']=origin_input[:, 0:cur_pos+1]
                        break


                print(self.tokenizer.decode(inputs["input_ids"][0]))

                prediction = inputs['input_ids'].cpu().numpy()
                all_predictions.extend(prediction)
                all_labels.extend(inputs["labels"].cpu().numpy())

                forms.extend(inputs['form'])
                stages.extend(inputs['stage'].tolist())

        filtered_preds, filtered_labels = [], []
        for pred, label in zip(all_predictions, all_labels):

            matches = np.where(pred==151643)
            if len(matches[0])>0:
                first_match_idx = matches[0][0]
                truncated_sample = pred[:first_match_idx]
            else:
                truncated_sample = pred
            # 找到 label 中所有非特殊 token 的位置
            valid_positions_for_label = [i for i, token in enumerate(label) if token not in self.special_id]
            valid_positions_for_predict = [i for i,token in enumerate(truncated_sample) if token not in self.special_id]
            # 从 predictions 和 labels 中提取对应的 token
            filtered_preds.append([truncated_sample[i] for i in valid_positions_for_predict])
            filtered_labels.append([label[i] for i in valid_positions_for_label])
        
        #将所有的id转化为文本
        str_predictions = self.tokenizer.batch_decode(filtered_preds)
        str_labels = self.tokenizer.batch_decode(filtered_labels)

        # forms = [d['form'] for d in dataloader.dataset][:sample_num]
        # stages = [d['stage'] for d in dataloader.dataset][:sample_num]
        pred_extra = {'forms': forms, 'stages': stages}
        avg_loss = np.mean(all_losses) if all_losses else None

        with open('eval_output_7B.pkl','wb') as f:
            pickle.dump(EvalLoopOutput(predictions=all_predictions, label_ids=all_labels,
                               metrics=avg_loss, num_samples=sample_num
                               ,pred_extra=pred_extra),f)


        return EvalLoopOutput(predictions=str_predictions, label_ids=str_labels,
                               metrics=avg_loss, num_samples=sample_num
                               ,pred_extra=pred_extra)  
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.generate(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        with open('eval_output_str.pkl','wb') as f:
            pickle.dump(output,f)
        # 调用 compute_metrics
        metrics = self.compute_metrics(output)
        print(metrics)
        # metrics.update(output.metrics)
        self.log(metrics)
        return metrics
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     outputs = model(**inputs)
    #     labels = inputs["labels"]
    #     stage = inputs["stage"]

    #     # 定义 stage 权重
    #     stage_weights = {1: 1, 2: 4, 3: 4, 4: 1}
    #     total_loss = 0
    #     total_weight = sum(stage_weights.values())

    #     # # 将 logits 和 labels 移到 GPU（如果需要）
    #     # logits = outputs.logits
    #     # labels = labels.to(logits.device)
    #     stage = torch.tensor(stage, dtype=torch.long, device=labels.device)  # 转换为张量

    #     for stage_value, weight in stage_weights.items():
    #         # 创建布尔掩码
    #         mask = stage == stage_value  # 使用张量的逐元素比较

    #         # 如果该 stage 有数据
    #         if mask.any():
    #             stage_outputs = outputs.logits[mask].view(-1, outputs.logits.size(-1))
    #             stage_labels = labels[mask].view(-1)
    #             stage_loss = self.loss_fn(stage_outputs, stage_labels)

    #             # 加权累加损失
    #             total_loss += weight * stage_loss

    #     # 计算平均损失
    #     loss = total_loss / total_weight
    #     return (loss, outputs) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     labels = inputs["labels"]
    #     stage = inputs["stage"]

    #     # 定义 stage 权重
    #     stage_weights = {1: 1, 2: 4, 3: 4, 4: 1}
    #     total_weight = sum(stage_weights.values())

    #     # 创建权重张量，与输入 batch 对齐
    #     weight_tensor = torch.tensor(
    #         [stage_weights[s] for s in stage], device=logits.device, dtype=torch.float
    #     )

    #     # 使用权重计算加权损失
    #     loss = self.loss_fn(
    #         logits.view(-1, logits.size(-1)),  # 展平 logits
    #         labels.view(-1),                   # 展平 labels                  # 不进行自动平均
    #     )

    #     # 按照权重计算加权平均
    #     weighted_loss = (loss * weight_tensor).mean() / total_weight
    #     return (weighted_loss, outputs) if return_outputs else loss

    def custom_compute_metrics(self,eval_pred: EvalLoopOutput) -> Dict[str, Any]:
        """
        针对 stages 为 1 或 2 的样本，计算 BLEU 和 ROUGE 指标。
        Args:
            eval_pred (EvalPrediction): 包含 predictions 和 labels，以及附加信息 pred_extra。
        
        Returns:
            Dict[str, Any]: BLEU 和 ROUGE 指标结果字典。
        """
        # 解析预测和标签
        labels =  eval_pred.label_ids
        stages = eval_pred.pred_extra['stages']        
        # 解析附加信息
        # 筛选 stages 为 1 
        stage1_indices = [i for i, stage in enumerate(stages) if stage in [1]]
        if  len(stage1_indices) >=1:
            # 提取对应的预测和标签
            stage1_labels = [labels[i] for i in stage1_indices]
            stage1_metrics = open_question_metrics([eval_pred.predictions[i] for i in stage1_indices], 
                                                   stage1_labels,self.special_id)

        #筛选出stage为2的样本
        stage2_indices = [i for i, stage in enumerate(stages) if stage in [2]]
        if len(stage2_indices) >=1:
            # 提取对应的预测和标签
            stage2_labels = [labels[i] for i in stage2_indices]
            stage2_metrics = closed_question_metrics( [eval_pred.predictions[i] for i in stage2_indices],
                                                     stage2_labels,self.special_id)

        #筛选出stage为3的样本
        stage3_indices = [i for i, stage in enumerate(stages) if stage in [3]]
        if  len(stage3_indices)>=1 :
            # 提取对应的预测和标签
            stage3_labels = [labels[i] for i in stage3_indices]
            stage3_metrics = closed_question_metrics( [eval_pred.predictions[i] for i in stage3_indices], 
                                                     stage3_labels,self.special_id)

        #筛选出stage为4的样本
        stage4_indices = [i for i, stage in enumerate(stages) if stage in [4]]
        if len(stage4_indices) >=1:
            # 提取对应的预测和标签
            stage4_labels = [labels[i] for i in stage4_indices]
            stage4_metrics = open_question_metrics([eval_pred.predictions[i] for i in stage4_indices],
                                                    stage4_labels,self.special_id)
        
        #合并存在的指标
        metrics = {}
        if stage1_indices:
            metrics.update({f"stage1_{k}": v for k, v in stage1_metrics.items()})
        if stage2_indices:
            metrics.update({f"stage2_{k}": v for k, v in stage2_metrics.items()})
        if stage3_indices:
            metrics.update({f"stage3_{k}": v for k, v in stage3_metrics.items()})
        if stage4_indices:
            metrics.update({f"stage4_{k}": v for k, v in stage4_metrics.items()})


        return metrics

