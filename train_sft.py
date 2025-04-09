from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from dataset.dataset import TsQaDataset,DataCollator
import argparse
from models.TimeLanguageModel import TLMConfig, TLM
import os
import wandb
from EXP.exp_instruct import Exp_Instruct
# # 限制只使用 GPU 0,debug模式
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# os.environ["RANK"] = "-1"
# os.environ["WORLD_SIZE"] = "1"

# # 启用异常检测
# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    #读取args
    parser = argparse.ArgumentParser(description='Mutimodal SFT')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    #TsEncoder  settings
    parser.add_argument('--model', type=str, required=False, default='TimeSeriesEncoder',
                        help='model name')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--load_ts_encoder', type=str, default='save/pretrain_ts/checkpoint-590', help='load ts_encoder')

    #TT-Former setting
    parser.add_argument('--tt_d_model', type=int, default=896, help='dimension of TT model')
    parser.add_argument('--tt_n_heads', type=int, default=16, help='num of TT heads')
    parser.add_argument('--tt_layers', type=int, default=2, help='num of TT layers')
    parser.add_argument('--tt_dropout', type=float, default=0.1, help='dropout for TT model')
    parser.add_argument('--prefix_num', type=int, default=25, help='number of prefixes')

    #LLM setting
    parser.add_argument('--llm_model_path', type=str, default='LLM/Qwen2.5-0.5B-Instruct', help='LLM model path')

    #Pretrain settings
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain mode')
    parser.add_argument('--min_mask_ratio', type=float, default=0.7, help='minimum mask ratio')
    parser.add_argument('--max_mask_ratio', type=float, default=0.8, help='maximum mask ratio')

    # Training arguments
    parser.add_argument('--do_train', type=bool, default=True, help='whether to do training')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='batch size per device during training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help='batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')

    #Efficiency settings
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use 16-bit (mixed) precision')
    parser.add_argument('--dataloader_pin_memory', type=bool, default=True, help='pin memory in data loader')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of subprocesses to use for data loading')

    #logging settings
    parser.add_argument('--output_dir', type=str, default='save/sft_v2_impadpre', help='output directory')
    parser.add_argument('--save_steps', type=int, default=500, help='save checkpoint every X updates steps')
    
    parser.add_argument('--save_total_limit', type=int, default=2, help='limit the total amount of checkpoints')
    parser.add_argument('--logging_steps', type=int, default=50, help='log every X updates steps')
    parser.add_argument('--eval_steps', type=int, default=3000, help='eval every X updates steps')

    parser.add_argument('--report_to', type=str, default='wandb', help='report results to')
    parser.add_argument('--mode', type=str, default='train', help='inference or train')
    parser.add_argument('--eval_stragy',type=str,default="no",help='The evaluation strategy to adopt during training')


    args = parser.parse_args()


    ##Model setting
    tlmconfig = TLMConfig(llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct',freeze_ts_model=True,
                          ts_pad_num=args.prefix_num)
    # model = TLM(tlmconfig,args).cuda()
    # print(model)
    # print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    ts_past_train = 'dataset/dataset_processing/data_merged_new.h5'
    qa_past_train = 'dataset/dataset_processing/train_sw3000.jsonl'
    
    ts_path_test = 'dataset/dataset_processing/data_merged_new.h5'
    qa_path_test = 'dataset/dataset_processing/test_sw3000.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(tlmconfig.llm_model_path)
    processor = AutoProcessor.from_pretrained(tlmconfig.llm_model_path)
    train_dataset = TsQaDataset(ts_past_train, qa_past_train, 
                          tokenizer, processor, tlmconfig,sft=True)
    test_dataset = TsQaDataset(ts_path_test, qa_path_test,
                            tokenizer, processor, tlmconfig)
    data_collator = DataCollator(tokenizer)


    # print(os.environ)


    # wandb.init(project="TSLLM", name="pandalin")

    Trainer = Exp_Instruct(args, train_dataset=train_dataset, eval_dataset=test_dataset,tlm_config=tlmconfig)
    Trainer.load_model('/dataYYF/dataWX/SJ/Time-QA/save/sft_v2_impadpre/checkpoint-22890')
    # Trainer.train(resume_from_checkpoint=True)
    Trainer.evaluate()
    # # Trainer.predict(test_dataset)
    # Trainer.save_model()
    # Trainer.save_state()

    
    

    
    