
import torch
import torch.nn as nn
from transformers import AutoModel

class HateXplainMultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout_rate=0.1, use_fusion=True):
        super(HateXplainMultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_fusion = use_fusion
        
        # Head 1: Sequence Classification (3 labels)
        hidden_dim = self.bert.config.hidden_size
        if self.use_fusion:
            # Pooled + Rationale
            self.class_classifier = nn.Linear(hidden_dim * 2, num_labels)
        else:
            # Just Pooled
            self.class_classifier = nn.Linear(hidden_dim, num_labels)
        
        # Head 2: Token Classification (2 labels: 0=Safe, 1=Toxic)
        self.token_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state # (Batch, SeqLen, Hidden)
        pooled_output = outputs.pooler_output # (Batch, Hidden) -> For CLS
        
        # Token Logits (Explanation Head)
        sequence_output_drop = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output_drop)
        
        if self.use_fusion:
            # --- Explanation-Guided Fusion ---
            # 1. Calculate Toxic Probabilities for each token
            token_probs = torch.softmax(token_logits, dim=-1) # (Batch, SeqLen, 2)
            toxic_probs = token_probs[:, :, 1].unsqueeze(-1)  # (Batch, SeqLen, 1) - Prob of being toxic
            
            # 2. Rationale Vector: Weighted Average of Token Embeddings
            # Avoid division by zero
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic # (Batch, Hidden)
            
            # 3. Concatenate [CLS] + Rationale
            combined_features = torch.cat((pooled_output, rationale_vector), dim=1)
            features = combined_features
        else:
            features = pooled_output
        
        # Classification Logits
        features = self.dropout(features)
        class_logits = self.class_classifier(features)
        
        return class_logits, token_logits
