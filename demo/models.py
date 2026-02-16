import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, AutoModel

class RATeDMultiTask(RobertaPreTrainedModel):
    """
    Unified model class for both VN and EN demo.
    Supports 3-class token labeling (BIO) and Fusion.
    """
    def __init__(self, config, use_fusion=True):
        super().__init__(config)
        self.use_fusion = use_fusion
        
        # Load pre-trained Backbone (Shared Encoder)
        self.roberta = AutoModel.from_config(config)
        
        # Classification Head (Sentence Level)
        input_dim = config.hidden_size * 2 if use_fusion else config.hidden_size
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(input_dim, config.num_labels)
        
        # Token Classification Head (BIO Tagging)
        self.token_dropout = nn.Dropout(config.hidden_dropout_prob)
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
        alpha=1.0 # Not used in inference but kept for signature
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )

        sequence_output = outputs[0]  # (Batch, Seq_Len, Hidden)
        pooled_output = outputs[1]    # (Batch, Hidden) - [CLS] token representation
        attentions = outputs.attentions

        # --- Task 2: Token Labeling ---
        token_output = self.token_dropout(sequence_output)
        token_logits = self.token_classifier(token_output) 
        
        # --- Explanation-Guided Fusion ---
        if self.use_fusion:
            token_probs = torch.softmax(token_logits, dim=-1)
            # BIO Tagging: Toxic = B + I
            toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)
            
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic
            
            combined_features = torch.cat((pooled_output, rationale_vector), dim=1)
            cls_input = combined_features
        else:
            cls_input = pooled_output
            
        # --- Task 1: Classification ---
        cls_output = self.cls_dropout(cls_input)
        cls_logits = self.cls_classifier(cls_output)

        return {
            'cls_logits': cls_logits,
            'token_logits': token_logits,
            'attentions': attentions
        }
