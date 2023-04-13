import torch

import torch.nn as nn
from torch.autograd import Variable
from bert import BertEmbedding
from config_model import IntentClassifier, SlotClassifier
from torchcrf import CRF


class LSTM_CRF(nn.Module):
    def __init__(self, args, num_intent_labels, num_slot_labels):
        super(LSTM_CRF, self).__init__()

        self.device = args.device
        self.embeddings = BertEmbedding(args)
        self.args = args

        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        
        self.hidden_size = args.hidden_size // 2
        self.num_layers = 2 

        self.lstm = nn.LSTM(input_size=self.embeddings.output_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True,dropout = 0.25)
        
        
        self.intent_classifier = IntentClassifier(self.hidden_size, self.num_intent_labels, 0.25)
        
        self.slot_classifier = SlotClassifier(
            self.hidden_size,
            self.num_slot_labels,
            args.bert_length,
            self.hidden_size // 2,
            0.25,
        )
        if args.use_crf == True:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)


    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        # print(embeddings)
        h_1 = Variable(torch.zeros(
            self.num_layers, embeddings.size(0), self.hidden_size).to(self.device))
         
        
        c_1 = Variable(torch.zeros(
            self.num_layers, embeddings.size(0), self.hidden_size).to(self.device))
       
        _, (hn, cn) = self.lstm(embeddings, (h_1, c_1))

        y = hn.view(-1, self.hidden_size)
        
        final_state_lstm = hn.view(self.num_layers, embeddings.size(0), self.hidden_size)[-1]
        logits_intent = self.intent_classifier(final_state_lstm)
        preds_intent = torch.argmax(logits_intent, dim=-1)


        logits_slot = self.slot_classifier(final_state_lstm)

        # """
        #     emissions (Tensor) – Emission score tensor of size (seq_length, batch_size, num_tags) 
        #     if batch_first is False, 
        #     (batch_size, seq_length, num_tags) otherwise.
        # """
        preds_slot = None
        if self.args.use_crf == True:
            # emissions = (batch_size)
            loss_slot = -1 * self.crf(logits_slot, 
                                   inputs['target_slots'].to(self.device),
                                   inputs['attention_mask'].byte().to(self.device), reduction="mean") #check xem trong embedding có attention mask ko, nếu ko phải sang bert return
            preds_slot = torch.Tensor(self.crf.decode(logits_slot))
        else: preds_slot = torch.argmax(logits_slot, dim=-1)
        # return logits_slot, preds_slot
        return logits_intent, preds_intent, logits_slot, preds_slot, loss_slot

