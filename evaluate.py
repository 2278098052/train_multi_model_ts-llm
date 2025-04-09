from typing import Dict, List, Any, NamedTuple, Optional, Tuple, Union
from utils.metrics import open_question_metrics,closed_question_metrics,compute_rul
import numpy as np
import pickle
from transformers import AutoTokenizer

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    pred_extra: Optional[Dict[str, Any]] = None


def custom_compute_metrics(self,eval_pred: EvalLoopOutput) -> Dict[str, Any]:
        """
        针对 stages 为 1 或 2 的样本，计算 BLEU 和 ROUGE 指标。
        Args:
            eval_pred (EvalPrediction): 包含 predictions 和 labels，以及附加信息 pred_extra。
        
        Returns:
            Dict[str, Any]: BLEU 和 ROUGE 指标结果字典。
        """
        # 解析预测和标签
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        forms = eval_pred.pred_extra['forms']
        stages = eval_pred.pred_extra['stages']        
        # 解析附加信息
        stages = eval_pred.pred_extra['stages']  # 假设 stages 是一个列表，对应每个样本的阶段
        
        # 筛选 stages 为 1 
        stage1_indices = [i for i, stage in enumerate(stages) if stage in [1]]
        if  len(stage1_indices) >=1:
            # 提取对应的预测和标签
            stage1_predictions = [predictions[i] for i in stage1_indices]
            stage1_labels = [labels[i] for i in stage1_indices]
            stage1_metrics = open_question_metrics(stage1_predictions, stage1_labels,self.special_id)

        #筛选出stage为2的样本
        stage2_indices = [i for i, stage in enumerate(stages) if stage in [2]]
        if len(stage2_indices) >=1:
            # 提取对应的预测和标签
            stage2_predictions = [predictions[i] for i in stage2_indices]
            stage2_labels = [labels[i] for i in stage2_indices]
            stage2_metrics = closed_question_metrics(stage2_predictions, stage2_labels,self.special_id)
        # for i in range(len(stage2_predictions)):
        #    if stage2_predictions[i] != stage2_labels[i]:
        #       print(stage2_predictions[i],stage2_labels[i])

        #筛选出stage为3的样本
        stage3_indices = [i for i, stage in enumerate(stages) if stage in [3]]
        if  len(stage3_indices)>=1 :
            # 提取对应的预测和标签
            stage3_predictions = [predictions[i] for i in stage3_indices]
            stage3_labels = [labels[i] for i in stage3_indices]
            stage3_metrics = compute_rul(stage3_predictions, stage3_labels)
        for i in range(len(stage3_predictions)):
           if stage3_predictions[i] != stage3_labels[i]:
              print(stage3_predictions[i],stage3_labels[i])
        


        #筛选出stage为4的样本
        stage4_indices = [i for i, stage in enumerate(stages) if stage in [4]]
        if len(stage4_indices) >=1:
            # 提取对应的预测和标签
            stage4_predictions = [predictions[i] for i in stage4_indices]
            stage4_labels = [labels[i] for i in stage4_indices]
            stage4_metrics = open_question_metrics(stage4_predictions, stage4_labels,self.special_id)
        
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

        print(metrics)
        return metrics


class Evaluator:
    def __init__(self, special_id):
        self.special_id = special_id
    custom_compute_metrics = custom_compute_metrics

    def process_output(self,output,tokenizer):

        all_predictions = output.predictions
        all_labels = output.label_ids

        # all_predictions = tokenizer.batch_decode(all_predictions, skip_special_tokens=False)
        # all_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=False)

        return EvalLoopOutput(predictions=all_predictions, label_ids=all_labels,
                                metrics=output.metrics, num_samples=output.num_samples
                                ,pred_extra=output.pred_extra)

if __name__ == "__main__":
    # Example usage
    with open('/dataYYF/dataWX/WYL/Time-QA/eval_output_str.pkl', 'rb') as f:
        eval_pred = pickle.load(f)
    llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    special_id = tokenizer.all_special_ids  
    common_punctuations = [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "\"", "'"]
    punctuation_ids = tokenizer.convert_tokens_to_ids(common_punctuations)
    special_id.extend(punctuation_ids)
    evaluator = Evaluator(special_id=special_id)

    eval_pred = evaluator.process_output(eval_pred,tokenizer)
    metrics = evaluator.custom_compute_metrics(eval_pred)
    print(metrics)