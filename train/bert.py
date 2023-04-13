import torch
from torch.nn import Module
from transformers import AutoModel


class BertEmbedding(Module):

    def __init__(self, parserargs):
        super(BertEmbedding, self).__init__()
        self.device = parserargs.device
        print('Init Bert')
        self.bert = AutoModel.from_pretrained(parserargs.bert_type)

        if not parserargs.update_bert:
            for params in self.bert.parameters():
                params.requires_grad = False
        self.output_size = 8 * self.bert.config.hidden_size

    def forward(self, inputs):

        BL = max(inputs['bert_length'])
        WL = max(inputs['word_length'])

        indices = inputs['indices'][:, :BL].to(self.device)

        attention_mask = inputs['attention_mask'][:, :BL].to(self.device)

        bert_outputs = self.bert(indices,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)

        bert_x = torch.concat(bert_outputs['hidden_states'][-8:], dim=-1)  # B x L x D
        transforms = inputs['transform_matrix'][:,:WL,:BL].to(self.device)

        embeddings = torch.bmm(transforms,bert_x)
        
        return embeddings


