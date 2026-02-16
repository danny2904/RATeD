import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, AutoModel

class HateXplainMultiTaskBIO(RobertaPreTrainedModel):
    def __init__(self, config, use_fusion=True):
        super().__init__(config)
        self.use_fusion = use_fusion
        
        # Load pre-trained Backbone (Shared Encoder)
        # For English, config.model_type is usually roberta
        self.roberta = AutoModel.from_config(config)
        
        # Classification Head (Sentence Level)
        # Input: [CLS] token embedding OR [CLS] + Rationale
        input_dim = config.hidden_size * 2 if use_fusion else config.hidden_size
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        # config.num_labels should be 3 for HateXplain
        self.cls_classifier = nn.Linear(input_dim, config.num_labels)
        
        # Token Classification Head (Word/Span Level)
        # Input: All token embeddings
        self.token_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output 3 classes per token: 0 (Normal), 1 (B-Toxic), 2 (I-Toxic)
        self.token_classifier = nn.Linear(config.hidden_size, 3) 
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,       # Classification labels (0=hate, 1=normal, 2=offensive)
        token_labels=None, # Token-level labels (0=normal, 1=B, 2=I)
        alpha=1.0,          # Weight for Token Loss
        use_consistency=True,
        lambda_const=0.1
    ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]  # (Batch, Seq_Len, Hidden)
        pooled_output = outputs[1]    # (Batch, Hidden) - [CLS] token representation

        # --- Task 2: Token Labeling for Rationale Extraction ---
        token_output = self.token_dropout(sequence_output)
        token_logits = self.token_classifier(token_output) # (Batch, Seq_Len, token_num_labels)
        token_num_labels = token_logits.size(-1)
        
        # --- Explanation-Guided Fusion ---
        if self.use_fusion:
            # Calculate Toxic Probabilities (Softmax) -> Use as Attention Weights
            token_probs = torch.softmax(token_logits, dim=-1) # (Batch, Seq_Len, tokens)
            
            # Handle both Binary (0=Normal, 1=Toxic) and BIO (0=Normal, 1=B, 2=I)
            if token_num_labels == 2:
                toxic_probs = token_probs[:, :, 1].unsqueeze(-1)
            else:
                # BIO Tagging
                toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)
            
            # Weighted Average of Token Embeddings based on Toxicity (Rationale Vector)
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic # (Batch, Hidden)
            
            # Concatenate [CLS] + Rationale Vector
            combined_features = torch.cat((pooled_output, rationale_vector), dim=1)
            cls_input = combined_features
        else:
            # No Fusion (Vanilla MTL)
            cls_input = pooled_output
            
        # --- Task 1: Classification ---
        cls_output = self.cls_dropout(cls_input)
        cls_logits = self.cls_classifier(cls_output) # (Batch, Num_Labels)

        total_loss = None
        
        if labels is not None and token_labels is not None:
            loss_fct_cls = nn.CrossEntropyLoss()
            loss_cls = loss_fct_cls(cls_logits.view(-1, self.config.num_labels), labels.view(-1))
            
            loss_fct_token = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padded tokens
            loss_token = loss_fct_token(token_logits.view(-1, token_num_labels), token_labels.view(-1))
            
            # Weighted Sum Multi-task Loss
            total_loss = loss_cls + (alpha * loss_token)
            
            # --- Consistency/Alignment Loss ---
            if use_consistency and lambda_const > 0:
                # Re-calculate toxic_probs for consistency logic
                token_probs = torch.softmax(token_logits, dim=-1)
                if token_num_labels == 2:
                    t_probs = token_probs[:, :, 1].unsqueeze(-1)
                else:
                    t_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)

                # English: label 1 is normal (Clean)
                clean_mask = (labels == 1)
                if clean_mask.sum() > 0:
                    clean_toxic_probs = t_probs[clean_mask].squeeze(-1) # (Num_Clean, Seq_Len)
                    loss_consistency = torch.mean(clean_toxic_probs)
                    total_loss += (lambda_const * loss_consistency)
            
        return {
            'loss': total_loss,
            'cls_logits': cls_logits,
            'token_logits': token_logits
        }
