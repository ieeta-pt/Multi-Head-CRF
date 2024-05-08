#from transformers import BertPreTrainedModel, BertForSequenceClassification, BertModel
import os
from typing import Optional, Union
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from layers.CRF import CRF
from itertools import islice


NUM_PER_LAYER = 16

class MultiHeadCRF(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def model_loss():
        return
    
    def __init__(self, config):
        super().__init__(config)
        self.number_of_layer_per_head = config.number_of_layer_per_head
        self.num_labels = config.num_labels
        self.heads = config.classes #expected an array of classes we are predicting
        self.bert = AutoModel.from_pretrained(config._name_or_path, config=config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        print(sorted(self.heads))
        for ent in  self.heads:
            for i in range(self.number_of_layer_per_head):
                setattr(self, f"{ent}_dense_{i}",            nn.Linear(config.hidden_size, config.hidden_size))
                setattr(self, f"{ent}_dense_activation_{i}", nn.GELU(approximate='none'))
            setattr(self, f"{ent}_classifier",       nn.Linear(config.hidden_size, config.num_labels))
            setattr(self, f"{ent}_crf",              CRF(num_tags=config.num_labels, batch_first=True))
            setattr(self, f"{ent}_reduction",        config.crf_reduction)
        self.reduction=config.crf_reduction

        if self.config.freeze == True:
            self.manage_freezing()
            
    def manage_freezing(self):
        for _, param in self.bert.embeddings.named_parameters():
            param.requires_grad = False
        
        num_encoders_to_freeze = self.config.num_frozen_encoder
        if num_encoders_to_freeze > 0:
            for _, param in islice(self.bert.encoder.named_parameters(), num_encoders_to_freeze*NUM_PER_LAYER):
                param.requires_grad = False
    
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
               ):
        # Default `model.config.use_return_dict´ is `True´
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # B S E 

        logits = {k:0 for k in self.heads}
        for ent in self.heads:
            for i in range(self.number_of_layer_per_head):
                dense_output = getattr(self, f"{ent}_dense_{i}")(sequence_output)
                dense_output = getattr(self, f"{ent}_dense_activation_{i}")(dense_output)
            logits[ent] = getattr(self, f"{ent}_classifier")(dense_output)
        #logits = self.classifier(sequence_output)
        loss = None
        if labels is not None: 
            # During train/test as we don't pass labels during inference
            
            # loss
            outputs = {k:0 for k in self.heads}
            for ent in self.heads:
                
                outputs[ent] = getattr(self, f"{ent}_crf")(logits[ent],labels[ent], reduction=self.reduction) 

            # print(outputs)
            return sum(outputs.values()), logits
        else: #running prediction?
            # decoded tags
            # NOTE: This gather operation (multiGPU) not work here, bc it uses tensors that are on CPU...
            outputs = {k:0 for k in self.heads}
            
            for ent in self.heads:
                outputs[ent] = torch.Tensor(getattr(self, f"{ent}_crf").decode(logits[ent]))
            return [outputs[ent] for ent in sorted(self.heads)]
