import sys
sys.path.append('/dataYYF/dataWX/WYL/Time-QA/')
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import argparse
from models.TimeSeriesEncoder import Model
from safetensors.torch import load_file
from models.TT_Former import TTformer
class TLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,llm_model_path = '/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                #  ts_model_path = '/home/user/Downloads/siglip-so400m-patch14-384',
                 freeze_ts_model = True,
                 ts_pad_num = 10,
                **kwargs):
        # self.ts_model_path = ts_model_path
        self.llm_model_path = llm_model_path
        self.freeze_ts_model = freeze_ts_model
        self.ts_pad_num = ts_pad_num
        super().__init__(**kwargs)

def compute_loss(logits, labels, loss_fn, stage):
        # 定义 stage 权重
        stage_weights = {1: 1, 2: 1, 3: 2, 4: 1}
        total_loss = 0
        total_weight = sum(stage_weights.values())

        # # 将 logits 和 labels 移到 GPU（如果需要）
        # logits = outputs.logits
        # labels = labels.to(logits.device)
        stage = torch.tensor(stage, dtype=torch.long, device=labels.device)  # 转换为张量

        for stage_value, weight in stage_weights.items():
            # 创建布尔掩码
            mask = stage == stage_value  # 使用张量的逐元素比较

            # 如果该 stage 有数据
            if mask.any():
                stage_outputs = logits[mask].view(-1, logits.size(-1))
                stage_labels = labels[mask].view(-1)
                stage_loss = loss_fn(stage_outputs, stage_labels)

                # 加权累加损失
                total_loss += weight * stage_loss

        # 计算平均损失
        loss = total_loss / total_weight
        return loss
        
class TLM(PreTrainedModel):
    config_class = TLMConfig
    def __init__(self, config,ts_config):
        super().__init__(config)
        self.config = config
        # self.mode = self.config.mode
        # self.ts_model = AutoModel.from_pretrained(self.config.ts_model_path)
        # self.processor = AutoProcessor.from_pretrained(self.config.ts_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        ts_config.llm_d_model = self.llm_model.config.hidden_size

        self.ts_encoder = Model(ts_config)
        # if ts_config.load_ts_encoder is not None:
        #     ts_encoder_weight = load_file(ts_config.load_ts_encoder + '/model.safetensors')
        #     self.ts_encoder.load_state_dict(ts_encoder_weight)
        #     print('load ts_encoder from', ts_config.load_ts_encoder)
        self.ts_encoder.cuda()
        self.ttformer = TTformer(ts_config)
        if not isinstance(self.ts_encoder, nn.Module):
            self.ts_encoder = nn.Module(self.ts_encoder)        
        # self.linear1 = nn.Linear(self.ts_model.config.ts_config.hidden_size*4, self.llm_model.config.hidden_size)
        # self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_ts_model:
            print('freeze Time Series Encoder')
            for param in self.ts_encoder.parameters():
                param.requires_grad = False
        print('freeze LLM')
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        self.ts_project = nn.Linear(ts_config.d_model,self.llm_model.config.hidden_size )
        self.fusion_project = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        # self.fusion_project = nn.Linear(ts_config.tt_d_model, self.llm_model.config.hidden_size)
    def forward(self, input_ids, labels, ts_values,stage=None,form=None, attention_mask=None,mode='train'):
    
        query_embeds = self.llm_model.get_input_embeddings()(input_ids)
        ts_embeds = self.ts_encoder(ts_values).logits
        ts_embeds = self.ts_project(ts_embeds)
        tt_embeds = self.ttformer(query_embeds,ts_embeds,stage)
        
        # text_embeds = text_embeds.to(tt_embeds.dtype)
        tt_embeds = self.fusion_project(tt_embeds)
        inputs_embeds = self.merge_input_ids_with_ts_features(tt_embeds, query_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        if mode=='train':
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = compute_loss(logits, labels, loss_fct, stage)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
                )
            return CausalLMOutputWithPast(loss=loss,logits=logits)
        elif mode=='inference':
            return logits

    #融合过程，出现了什么事情
    def merge_input_ids_with_ts_features(self, ts_features, inputs_embeds, input_ids):
        
        num_tss, num_ts_patches, embed_dim = ts_features.shape
        batch_indices, ts_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        inputs_embeds[batch_indices, ts_indices] = ts_features.view(-1, embed_dim)
        
        return inputs_embeds
    

        
        
if __name__ == '__main__':

    #读取args
    parser = argparse.ArgumentParser(description='TLM Define')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    #TsEncoder  settings
    parser.add_argument('--model', type=str, required=False, default='TimeSeriesEncoder',
                        help='model name')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--load_ts_encoder', type=str, default='save/pretrain_ts/checkpoint-500', help='load ts_encoder')

    #TTformer settings
    parser.add_argument('--tt_d_model', type=int, default=896, help='dimension of TT model')
    parser.add_argument('--tt_n_heads', type=int, default=8, help='num of TT heads')
    parser.add_argument('--tt_layers', type=int, default=6, help='num of TT layers')
    parser.add_argument('--tt_dropout', type=float, default=0.1, help='dropout for TT model')
    parser.add_argument('--prefix_num', type=int, default=25, help='number of prefixes')

    # parser.add_argument('--pretrain', type=bool, default=True, help='pretrain mode')
    # parser.add_argument('--min_mask_ratio', type=float, default=0.7, help='minimum mask ratio')
    # parser.add_argument('--max_mask_ratio', type=float, default=0.8, help='maximum mask ratio')

    # Ts_encoder = Model(ts_args)
    args = parser.parse_args()


    tlmconfig = TLMConfig(llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct')
    model = TLM(tlmconfig,args)
    # #加载语句：
    # model =TLM.from_pretrained("save/pretrain_new/checkpoint-5122", config=tlmconfig, ts_config=ts_args)
    model.cuda()

    tokenizer = model.tokenizer
    
    ts_pad_token_id = tokenizer('<|image_pad|>')['input_ids'][0]
    input_ids = torch.tensor([[tokenizer.pad_token_id] + [ts_pad_token_id] * model.config.ts_pad_num + [tokenizer.pad_token_id, tokenizer.pad_token_id]]).cuda()
    labels = torch.tensor([[tokenizer.pad_token_id] + [ts_pad_token_id] * model.config.ts_pad_num + [tokenizer.pad_token_id, tokenizer.pad_token_id]]).cuda()
    ts_values = torch.randn(1, 600, 33).cuda()

    outputs = model(input_ids, labels, ts_values)


    
    print(outputs)
