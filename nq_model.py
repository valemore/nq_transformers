'''
Implements the baseline model on top of BERT described here: https://arxiv.org/abs/1901.08634
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # Following import causes annoying warning
from transformers.modeling_bert import (BertModel, BertPreTrainedModel)

class NQBert(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **type_scores**: ``torch.FloatTensor`` of shape ``(batch_size, 3,)``
            Scores for question type (null instance, long answer only, short answer, yes, no) (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] 
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)  
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet
    """
    def __init__(self, config):
        super(NQBert, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        assert(config.num_labels == 2)
        self.span_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.type_output = nn.Linear(config.hidden_size, 5)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, instance_types=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.span_outputs(sequence_output)
        start_logits, end_logits  = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        type_logits = self.type_output(pooled_output)

        outputs = (start_logits, end_logits, type_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None and instance_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(instance_types.size()) > 1:
                instance_types = instance_types.squeeze(-1) # I think this is needed - but have not trained on multi-GPU
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            type_loss_fct = CrossEntropyLoss()
            start_loss = span_loss_fct(start_logits, start_positions)
            end_loss = span_loss_fct(end_logits, end_positions)
            type_loss = type_loss_fct(type_logits, instance_types)
            total_loss = (start_loss + end_loss + type_loss) / 3
            total_loss = torch.unsqueeze(total_loss, 0) # For multi-gpu training
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, type_logits, (hidden_states), (attentions)
