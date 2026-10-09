import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='km3fend')#textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--dataset', default='ch')# en
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')
parser.add_argument('--mode', default='train', choices=['train', 'test'], help='运行模式：train=训练+测试；test=仅测试（需配合--load_ckpt）')
parser.add_argument('--load_ckpt', type=str, default='', help='若非空，则从该路径加载模型权重（.pkl/.pt）')
parser.add_argument('--semantic_num', type=int, default=7)
parser.add_argument('--emotion_num', type=int, default=7)
parser.add_argument('--style_num', type=int, default=2)
parser.add_argument('--lnn_dim', type=int, default=50)
parser.add_argument('--domain_num', type=int, default=3)
parser.add_argument('--con_weight', type=float, default=0.03,
                    help='Trust-weighted InfoNCE weight; 0 disables (kg_pack sets 0.03)')
parser.add_argument('--con_temperature', type=float, default=0.123)
parser.add_argument('--k_gate_strength', type=float, default=0.4)
parser.add_argument('--knowledge_mode', default='kg', choices=['kg', 'llm', 'none'],
                    help='kg=structured trust fusion; llm=legacy hard replace; none=ignore knowledge')
parser.add_argument('--use_kg_cross_attn', action='store_true', default=True,
                    help='KG mode: cross-attention refine knowledge with content tokens')
parser.add_argument('--no_kg_cross_attn', action='store_false', dest='use_kg_cross_attn',
                    help='Disable KG cross-attention (tau fusion only)')
parser.add_argument('--use_soft_domain', action='store_true', help='SLFEND-style soft domain routing blend')
parser.add_argument('--soft_domain_beta', type=float, default=0.2, help='Blend weight for soft domain vs memory att')
parser.add_argument('--soft_domain_loss_weight', type=float, default=0.1, help='Aux CE weight for soft domain head')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='BCE label smoothing (KG-MFEND, 0 to disable)')
parser.add_argument('--con_trust_threshold', type=float, default=0.3, help='Min tau for trust-weighted InfoNCE')
parser.add_argument('--bert_finetune', action='store_true', help='是否微调BERT/Roberta参数')
parser.add_argument('--bert_lr', type=float, default=2e-5, help='微调BERT时的学习率')
parser.add_argument('--auto_threshold', action='store_true', help='在验证集搜索最佳分类阈值并用于测试集')
parser.add_argument('--balanced_domain_sampler', action='store_true', help='训练阶段按领域均衡采样，提升低频领域学习强度')
parser.add_argument('--domain_reweight_loss', action='store_true', help='按领域样本频次重加权分类损失')
parser.add_argument('--domain_weight_alpha', type=float, default=0.7, help='领域重加权强度，越大越偏向低频领域')
parser.add_argument('--domain_weight_max_ratio', type=float, default=2.0, help='领域重加权最大倍数，防止过度牺牲高表现领域')
parser.add_argument('--domain_reweight_warmup_epochs', type=int, default=3, help='领域重加权预热轮数，前若干轮逐步开启弱域增强')
parser.add_argument('--focal_gamma', type=float, default=0.0, help='focal难例聚焦系数，0表示关闭')
parser.add_argument('--bert_unfreeze_layers', type=int, default=0, help='bert_finetune时仅解冻最后N层；0表示全量解冻')
parser.add_argument('--use_domain_adapter', action='store_true', help='启用领域特定适配器，让不同领域走不同优化分支')
parser.add_argument('--domain_adapter_scale', type=float, default=0.2, help='领域适配残差强度，建议 0.1~0.3')
parser.add_argument('--use_domain_moe', action='store_true', help='启用领域MoE分类头（领域专家+共享专家+路由）')
parser.add_argument('--moe_router_temp', type=float, default=1.0, help='MoE路由温度，越小越尖锐')
parser.add_argument('--moe_shared_weight', type=float, default=0.2, help='共享专家兜底权重，越大越保守')
parser.add_argument('--moe_confidence_scale', type=float, default=3.5, help='路由置信门控强度')
parser.add_argument('--moe_no_shared', action='store_true', help='MoE消融：关闭共享专家，仅使用领域专家')
parser.add_argument('--moe_soft_routing', action='store_true', help='MoE消融：软路由，按Memory注意力加权激活多个领域专家')
parser.add_argument('--moe_expert_min_weight', type=float, default=0.0, help='软路由专家激活阈值(0~1)，仅保留占比>=该值的专家并重新归一化；0表示不过滤；例0.2表示>=20%%才参与')
parser.add_argument('--moe_topk_experts', type=int, default=0, help='软路由Top-K激活专家数；0=全部激活（默认）；>0=仅保留路由权重最高的K个领域专家并重新归一化（对9领域设1~9）')
parser.add_argument('--use_bilinear_fusion', action='store_true', help='启用门控双线性融合：S\' = g*S + (1-g)*tau*(S⊙K_proj)，替代简单加性残差')
parser.add_argument('--use_multi_factor_trust', action='store_true', help='启用多因子信任评分：综合语义一致性、实体覆盖率、关系相关性、不确定性')
parser.add_argument('--use_triple_encoder', action='store_true', help='启用关系感知三元组编码器：保留三元组结构信息，动态选择最相关三元组')
parser.add_argument('--max_triples', type=int, default=8, help='三元组编码器支持的最大三元组数量')
parser.add_argument('--use_memory_bank', action='store_true', default=True, help='启用领域记忆库（Memory Bank），输出领域注意力权重')
parser.add_argument('--no_memory_bank', action='store_false', dest='use_memory_bank', help='关闭领域记忆库，使用均匀领域注意力')
parser.add_argument('--save_case_results', type=str, default='', help='若非空，则 test 阶段把每条样本的中间字段保存到该 CSV 路径')
parser.add_argument('--improve_pack', action='store_true', help='一键启用稳定增益策略（弱域增强+阈值搜索+后层微调）')
parser.add_argument('--kg_pack', action='store_true',
                    help='KG-MFEND 推荐：kg 融合 + label_smoothing + con_weight + 训练前自动生成 KG 知识')
parser.add_argument('--prepare_kg', action='store_true',
                    help='训练前若缺少 *_kg_knowledge.npy 则运行 scripts/prepare_kg_knowledge.py')
parser.add_argument('--skip_prepare_kg', action='store_true', help='与 --kg_pack 联用时跳过自动生成知识')
parser.add_argument(
    '--n_runs',
    type=int,
    default=1,
    help='连续训练次数；每次结束会各写一行到 docs/experiment_results.md（对齐论文多次平均时可设 10）',
)
parser.add_argument(
    '--seed_step',
    type=int,
    default=1,
    help='第 k 次运行的随机种子 = seed + (k-1)*seed_step；设为 0 则每次种子相同',
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.kg_pack:
    args.knowledge_mode = 'kg'
    args.use_kg_cross_attn = True
    if args.label_smoothing <= 0:
        args.label_smoothing = 0.1
    if args.con_weight <= 0:
        args.con_weight = 0.03
    if args.k_gate_strength <= 0:
        args.k_gate_strength = 0.4
    args.prepare_kg = not args.skip_prepare_kg

if args.improve_pack:
    args.balanced_domain_sampler = False
    args.domain_reweight_loss = True
    args.use_domain_adapter = True
    args.use_domain_moe = True
    if args.focal_gamma <= 0:
        args.focal_gamma = 1.0
    if args.domain_weight_max_ratio <= 1.0:
        args.domain_weight_max_ratio = 2.0
    if args.domain_reweight_warmup_epochs < 3:
        args.domain_reweight_warmup_epochs = 3
    args.auto_threshold = True
    args.bert_finetune = True
    if args.bert_unfreeze_layers <= 0:
        args.bert_unfreeze_layers = 4
    if args.early_stop < 5:
        args.early_stop = 5

from grid_search import Run
import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if args.dataset == 'en':
    root_path = './data/en/'
    category_dict = {
        "gossipcop": 0,
        "politifact": 1,
        "COVID": 2,
    }
elif args.dataset == 'ch':
    root_path = './data/ch/'
    if args.domain_num == 9:
        category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
        }
    elif args.domain_num == 6:
        category_dict = {
            "教育考试": 0,
            "灾难事故": 1,
            "医药健康": 2,
            "财经商业": 3,
            "文体娱乐": 4,
            "社会生活": 5,
        }
    elif args.domain_num == 3:
        category_dict = {
            "政治": 0,  #852
            "医药健康": 1,  #1000
            "文体娱乐": 2,  #1440
        }

def _kg_knowledge_ready(data_root: str) -> bool:
    for split in ('train', 'val', 'test'):
        pkl = os.path.join(data_root, f'{split}.pkl')
        kn = os.path.join(data_root, f'{split}_kg_knowledge.npy')
        if os.path.exists(pkl) and not os.path.exists(kn):
            return False
    return True


if args.prepare_kg and args.model_name == 'km3fend' and args.knowledge_mode != 'none':
    if not _kg_knowledge_ready(root_path):
        import subprocess
        import sys
        prep = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'prepare_kg_knowledge.py')
        print('[main] missing *_kg_knowledge.npy, running prepare_kg_knowledge.py ...', flush=True)
        subprocess.run([sys.executable, prep, '--dataset', args.dataset], check=True)
    else:
        print('[main] *_kg_knowledge.npy already present, skip prepare_kg', flush=True)

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}; domain_num: {}; knowledge_mode: {}'.format(
    args.lr, args.model_name, args.batchsize, args.epoch, args.gpu, args.domain_num, args.knowledge_mode))


config = {
        'use_cuda': torch.cuda.is_available(),
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'root_path': root_path,
        'weight_decay': 5e-5,
        'category_dict': category_dict,
        'dataset': args.dataset,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'semantic_num': args.semantic_num,
        'emotion_num': args.emotion_num,
        'style_num': args.style_num,
        'domain_num': args.domain_num,
        'lnn_dim': args.lnn_dim,#the number of cross-view representations
        'con_weight': args.con_weight,
        'con_temperature': args.con_temperature,
        'k_gate_strength': args.k_gate_strength,
        'knowledge_mode': args.knowledge_mode,
        'use_kg_cross_attn': args.use_kg_cross_attn,
        'use_soft_domain': args.use_soft_domain,
        'soft_domain_beta': args.soft_domain_beta,
        'soft_domain_loss_weight': args.soft_domain_loss_weight,
        'label_smoothing': args.label_smoothing,
        'con_trust_threshold': args.con_trust_threshold,
        'bert_finetune': args.bert_finetune,
        'bert_lr': args.bert_lr,
        'auto_threshold': args.auto_threshold,
        'balanced_domain_sampler': args.balanced_domain_sampler,
        'domain_reweight_loss': args.domain_reweight_loss,
        'domain_weight_alpha': args.domain_weight_alpha,
        'domain_weight_max_ratio': args.domain_weight_max_ratio,
        'domain_reweight_warmup_epochs': args.domain_reweight_warmup_epochs,
        'focal_gamma': args.focal_gamma,
        'bert_unfreeze_layers': args.bert_unfreeze_layers,
        'use_domain_adapter': args.use_domain_adapter,
        'domain_adapter_scale': args.domain_adapter_scale,
        'use_domain_moe': args.use_domain_moe,
        'moe_router_temp': args.moe_router_temp,
        'moe_shared_weight': args.moe_shared_weight,
        'moe_confidence_scale': args.moe_confidence_scale,
        'moe_no_shared': args.moe_no_shared,
        'moe_soft_routing': args.moe_soft_routing,
        'moe_expert_min_weight': args.moe_expert_min_weight,
        'moe_topk_experts': args.moe_topk_experts,
        'use_bilinear_fusion': args.use_bilinear_fusion,
        'use_multi_factor_trust': args.use_multi_factor_trust,
        'use_triple_encoder': args.use_triple_encoder,
        'max_triples': args.max_triples,
        'use_memory_bank': args.use_memory_bank,
        'save_case_results': args.save_case_results,
        'mode': args.mode,
        'load_ckpt': args.load_ckpt,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir
        }



if __name__ == '__main__':
    n = max(1, int(args.n_runs))
    for run_idx in range(n):
        current_seed = int(args.seed) + run_idx * int(args.seed_step)
        set_seed(current_seed)
        config['seed'] = current_seed
        if n > 1:
            print('=== run {}/{} seed={} ==='.format(run_idx + 1, n, current_seed))
        Run(config=config).main()
