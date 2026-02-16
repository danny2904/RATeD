import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel, AutoModel

class XLMRMultiTask(RobertaPreTrainedModel):
    def __init__(self, config, use_fusion=True):
        super().__init__(config)
        self.use_fusion = use_fusion
        
        # Load pre-trained Backbone (Shared Encoder)
        self.roberta = AutoModel.from_config(config)
        
        # Classification Head (Sentence Level)
        # Input: [CLS] token embedding OR [CLS] + Rationale
        input_dim = config.hidden_size * 2 if use_fusion else config.hidden_size
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(input_dim, config.num_labels)
        
        # Token Classification Head (Word/Span Level)
        # Input: All token embeddings
        self.token_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output 2 classes per token: 0 (Normal), 1 (Toxic)
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
        labels=None,       # Classification labels (Clean/Hate)
        token_labels=None, # Token-level labels (0/1 for each token)
        alpha=1.0,          # Weight for Token Loss
        use_consistency=True,
        lambda_const=0.1
    ):
        
        # By default, PhoBERT (Roberta) doesn't use token_type_ids, but we keep arg for compatibility
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

        # --- Task 2 Pre-calc: Token Labeling for Rationale Extraction ---
        token_output = self.token_dropout(sequence_output)
        token_logits = self.token_classifier(token_output) # (Batch, Seq_Len, 2)
        
        # --- Explanation-Guided Fusion ---
        if self.use_fusion:
            # Calculate Toxic Probabilities (Softmax) -> Use as Attention Weights
            token_probs = torch.softmax(token_logits, dim=-1) # (Batch, Seq_Len, 3)
            # Prob of being toxic = Prob(B-Toxic) + Prob(I-Toxic)
            toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)  # (Batch, Seq_Len, 1)
            
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
            loss_token = loss_fct_token(token_logits.view(-1, 3), token_labels.view(-1))
            
            # Weighted Sum Multi-task Loss
            total_loss = loss_cls + (alpha * loss_token)
            
            # --- Consistency/Alignment Loss ---
            if use_consistency and lambda_const > 0:
                # Re-calculate probs if not done by fusion block (could optimize but keeping clean logic)
                token_probs = torch.softmax(token_logits, dim=-1) # (Batch, Seq_Len, 3)
                toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)  # (Batch, Seq_Len, 1)

                clean_mask = (labels == 0)
                if clean_mask.sum() > 0:
                    clean_toxic_probs = toxic_probs[clean_mask].squeeze(-1) # (Num_Clean, Seq_Len)
                    loss_consistency = torch.mean(clean_toxic_probs)
                    total_loss += (lambda_const * loss_consistency)
            
        return {
            'loss': total_loss,
            'cls_logits': cls_logits,
            'token_logits': token_logits
        }
