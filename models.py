import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaClassificationHead

import logging
logger = logging.getLogger(__name__)

class RobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        self.label_token_list = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)
#         all_logits = prediction_mask_scores
        all_logits = F.softmax(prediction_mask_scores, dim=-1)

        if self.label_token_list is not None:
            logits = []
            for label in self.label_token_list:
                logits.append(torch.sum(all_logits[:, self.label_token_list[label]], 1).unsqueeze(-1))
            logits = torch.cat(logits, -1)

        loss = None
        if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.NLLLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss_fct(torch.log(logits.view(-1, logits.size(-1))), labels.view(-1))

        output = (all_logits,)
        if self.label_token_list is not None:
            output = ((logits,) + output)
        return ((loss,) + output) if loss is not None else output