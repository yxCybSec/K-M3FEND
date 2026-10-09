import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder


class DHEMModel(nn.Module):
    def __init__(self, emb_dim, semantic_num=7, emotion_num=7, style_num=2, 
                 lnn_dim=50, domain_num=9, dropout=0.2, bert_model='/home/ubuntu/.cache/huggingface/hub/models--hfl--chinese-bert-wwm-ext/snapshots/2a995a880017c60e4683869e817130d8af548486',
                 emotion_dim=235, style_dim=48, dataset='ch'):
        super().__init__()
        if dataset == 'en':
            self.bert = RobertaModel.from_pretrained(bert_model).requires_grad_(False)
        else:
            self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        self.domain_num = domain_num
        self.emotion_dim = emotion_dim
        self.style_dim = style_dim
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.semantic_expert_count = len(feature_kernel)
        
        self.semantic_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, 64, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for k in feature_kernel.keys()
        ])
        
        self.emotion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emotion_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 320),
                nn.ReLU()
            ) for _ in range(emotion_num)
        ])
        
        self.style_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(style_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 320),
                nn.ReLU()
            ) for _ in range(style_num)
        ])
        
        self.shared_semantic = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.shared_emotion = nn.Sequential(
            nn.Linear(emotion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 320),
            nn.ReLU()
        )
        
        self.shared_style = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 320),
            nn.ReLU()
        )
        
        self.semantic_gate = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.semantic_expert_count + 1),
            nn.Softmax(dim=1)
        )
        
        self.emotion_gate = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emotion_num + 1),
            nn.Softmax(dim=1)
        )
        
        self.style_gate = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, style_num + 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 320 + 320, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, content, content_masks, comments, comments_masks,
                content_emotion, comments_emotion, emotion_gap, style_feature, 
                label, category, domain_feature):
        batch_size = content.size(0)
        
        content_feat = self.bert(content, attention_mask=content_masks)[0]
        semantic = self._attention_pooling(content_feat, content_masks)
        
        semantic_gate_val = self.semantic_gate(semantic)
        emotion_gate_val = self.emotion_gate(semantic)
        style_gate_val = self.style_gate(semantic)
        
        semantic_feats = []
        for expert in self.semantic_experts:
            fea = expert(content_feat.permute(0, 2, 1)).squeeze(-1)
            semantic_feats.append(fea)
        semantic_feats = torch.stack(semantic_feats, dim=1)
        shared_sem_feat = self.shared_semantic(content_feat.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
        semantic_feats = torch.cat([semantic_feats, shared_sem_feat], dim=1)
        semantic_output = torch.sum(semantic_feats * semantic_gate_val.unsqueeze(-1), dim=1)
        
        emotion_feat = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        emotion_feats = []
        for expert in self.emotion_experts:
            fea = expert(emotion_feat)
            emotion_feats.append(fea)
        emotion_feats = torch.stack(emotion_feats, dim=1)
        shared_emo_feat = self.shared_emotion(emotion_feat).unsqueeze(1)
        emotion_feats = torch.cat([emotion_feats, shared_emo_feat], dim=1)
        emotion_output = torch.sum(emotion_feats * emotion_gate_val.unsqueeze(-1), dim=1)
        
        style_feats = []
        for expert in self.style_experts:
            fea = expert(style_feature)
            style_feats.append(fea)
        style_feats = torch.stack(style_feats, dim=1)
        shared_style_feat = self.shared_style(style_feature).unsqueeze(1)
        style_feats = torch.cat([style_feats, shared_style_feat], dim=1)
        style_output = torch.sum(style_feats * style_gate_val.unsqueeze(-1), dim=1)
        
        combined = torch.cat([semantic_output, emotion_output, style_output], dim=1)
        
        logits = self.classifier(combined)
        pred = torch.sigmoid(logits)
        
        return pred.squeeze(1)
    
    def _attention_pooling(self, features, masks):
        attn_weights = torch.ones(features.size(0), features.size(1), 1, device=features.device)
        attn_weights = attn_weights.masked_fill(masks.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        return torch.bmm(attn_weights.permute(0, 2, 1), features).squeeze(1)


class Trainer:
    def __init__(self, emb_dim, mlp_dims, use_cuda, lr, train_loader, dropout,
                 val_loader, test_loader, category_dict, weight_decay, save_param_dir,
                 semantic_num=7, emotion_num=7, style_num=2, lnn_dim=50, 
                 dataset='ch', early_stop=3, epoches=50):
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.use_cuda = use_cuda
        self.lr = lr
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.category_dict = category_dict
        self.save_param_dir = save_param_dir
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.early_stop = early_stop
        self.epoches = epoches
        self.domain_num = len(category_dict)
        
        emotion_dim = 235
        style_dim = 48
        if len(train_loader) > 0:
            sample_batch = next(iter(train_loader))
            # sample_batch是tuple: (content_ids, content_masks, comments_ids, comments_masks,
            #                       content_emotion, comments_emotion, emotion_gap, style_feature,
            #                       label, category, domain_feature)
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 8:
                emo_t = sample_batch[4]
                com_t = sample_batch[5]
                gap_t = sample_batch[6]
                sty_t = sample_batch[7]
                emo_dim = emo_t.shape[1] if len(emo_t.shape) > 1 else 0
                com_dim = com_t.shape[1] if len(com_t.shape) > 1 else 0
                gap_dim = gap_t.shape[1] if len(gap_t.shape) > 1 else 0
                emotion_dim = emo_dim + com_dim + gap_dim
                if len(sty_t.shape) > 1:
                    style_dim = sty_t.shape[1]
            elif isinstance(sample_batch, dict):
                if 'content_emotion' in sample_batch:
                    emo_dim = sample_batch['content_emotion'].shape[1] if len(sample_batch['content_emotion'].shape) > 1 else 0
                    com_dim = sample_batch['comments_emotion'].shape[1] if 'comments_emotion' in sample_batch and len(sample_batch['comments_emotion'].shape) > 1 else 0
                    gap_dim = sample_batch['emotion_gap'].shape[1] if 'emotion_gap' in sample_batch and len(sample_batch['emotion_gap'].shape) > 1 else 0
                    emotion_dim = emo_dim + com_dim + gap_dim
                if 'style_feature' in sample_batch and len(sample_batch['style_feature'].shape) > 1:
                    style_dim = sample_batch['style_feature'].shape[1]
        
        bert_model = '/home/ubuntu/.cache/huggingface/hub/models--hfl--chinese-bert-wwm-ext/snapshots/2a995a880017c60e4683869e817130d8af548486' if dataset == 'ch' else 'roberta-base'
        self.model = DHEMModel(
            emb_dim=emb_dim,
            semantic_num=semantic_num,
            emotion_num=emotion_num,
            style_num=style_num,
            lnn_dim=lnn_dim,
            domain_num=self.domain_num,
            dropout=dropout,
            bert_model=bert_model,
            emotion_dim=emotion_dim,
            style_dim=style_dim,
            dataset=dataset
        )
        
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def train(self, logger=None):
        os.makedirs(self.save_param_dir, exist_ok=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoches)
        
        recorder = Recorder(self.early_stop)
        best_metric = {'metric': 0}
        best_model_path = None
        
        for epoch in range(self.epoches):
            self.model.train()
            loss_sum = 0
            loss_averager = Averager()
            
            pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
            for batch in pbar:
                batch_data = data2gpu(batch, self.use_cuda)
                
                pred = self.model(
                    batch_data['content'], batch_data['content_masks'],
                    batch_data['comments'], batch_data['comments_masks'],
                    batch_data['content_emotion'], batch_data['comments_emotion'],
                    batch_data['emotion_gap'], batch_data['style_feature'],
                    batch_data['label'], batch_data['category'],
                    batch_data['domain_feature']
                )
                
                loss = F.binary_cross_entropy(pred, batch_data['label'].float())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_averager.add(loss.item())
                loss_sum += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            avg_loss = loss_sum / max(1, len(self.train_loader))
            
            val_metrics = self._validate()
            val_metric = val_metrics['metric']
            
            if logger:
                logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, val_metric={val_metric:.4f}')
            
            print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, val_metric={val_metric:.4f}')
            
            status = recorder.add(val_metrics)
            
            if status == 'save':
                best_metric = val_metrics
                best_model_path = os.path.join(self.save_param_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f'Saved best model with metric: {val_metric:.4f}')
            elif status == 'esc':
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return best_metric, best_model_path
    
    def _validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_categories = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch_data = data2gpu(batch, self.use_cuda)
                
                pred = self.model(
                    batch_data['content'], batch_data['content_masks'],
                    batch_data['comments'], batch_data['comments_masks'],
                    batch_data['content_emotion'], batch_data['comments_emotion'],
                    batch_data['emotion_gap'], batch_data['style_feature'],
                    batch_data['label'], batch_data['category'],
                    batch_data['domain_feature']
                )
                
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_data['label'].cpu().numpy())
                all_categories.append(batch_data['category'].cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_categories = np.concatenate(all_categories)
        
        res = metrics(all_labels, all_preds, all_categories, self.category_dict)
        return res
    
    def test(self):
        best_path = os.path.join(self.save_param_dir, 'best_model.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location='cpu'))
            if self.use_cuda:
                self.model = self.model.cuda()
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_categories = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch_data = data2gpu(batch, self.use_cuda)
                
                pred = self.model(
                    batch_data['content'], batch_data['content_masks'],
                    batch_data['comments'], batch_data['comments_masks'],
                    batch_data['content_emotion'], batch_data['comments_emotion'],
                    batch_data['emotion_gap'], batch_data['style_feature'],
                    batch_data['label'], batch_data['category'],
                    batch_data['domain_feature']
                )
                
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_data['label'].cpu().numpy())
                all_categories.append(batch_data['category'].cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_categories = np.concatenate(all_categories)
        
        res = metrics(all_labels, all_preds, all_categories, self.category_dict)
        print(f'Test Results: {res}')
        return res
