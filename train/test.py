# import os
# cached_features_file = os.path.join(
#         "./PhoATIS",
#         "cached_{}_{}_{}_{}".format(
#             'train', "word-level", list(filter(None, "vinai/phobert-base".split("/"))).pop(), 50
#         ),
#     )

# print(cached_features_file)

from transformers import AutoTokenizer, AutoModel
from config.parser_argument import parserargs

# tokenizer = AutoTokenizer.from_pretrained(parserargs.bert_type)
# a = tokenizer.tokenize('12^7*)')
# if not a:
#     print(a)
# else:
#     print('hi',a)




"""        
        transform_matrix = np.zeros((self.WML,self.BML,), dtype=np.float32)
        
        all_pieces = [self.CLS]
        transform_matrix[0,len(all_pieces)-1] = 1.0
        all_spans = []
        
        for idx, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)

            pieces = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(pieces) == 0:
                pieces = [self.UNK]
            start = len(all_pieces)
            all_pieces += pieces
            end = len(all_pieces)

            # if end > self.WML - 1: # except special tokens: SEP (token CLS is existed in all_pieces)
            #     all_pieces = all_pieces[:self.WML - 1]
                
            all_spans.append([start, end])

            if len(pieces) != 0:
                piece_num = len(pieces)
                mean_matrix = np.full((piece_num), 1.0/piece_num)
                transform_matrix[idx+1,start:end] = mean_matrix
            
                
        all_pieces.append(self.SEP)

        # pad_length = self.WML - len(all_pieces)

        # all_pieces += [self.PAD] * pad_length
        cls_text_sep_length = len(all_pieces)
        transform_matrix[len(words),cls_text_sep_length-1] = 1.0
        assert len(all_pieces) <= self.BML
        
        pad_len = self.BML - len(all_pieces)
        all_pieces += [self.PAD] * pad_len
        attention_mask = [1.0] * cls_text_sep_length + [0.0] * pad_len
        assert len(all_pieces) == self.BML
    """

# import torch
# x = torch.Tensor([[1,2], [3,4], [5, 6]])
# y = torch.Tensor([[1,2], [3,4], [5, 6]])
# print(x.size())
# print(x)
# print(x.view(-1, 6))

# x = torch.unsqueeze(x, 1)
# print(x.size())
# print(x)
# a = x.expand(-1, 3, -1)
# print(a.size())
# print(a)

# tmp = torch.cat((y, a), dim=2)
# print(tmp.size())
# print(tmp)
# from torch.nn import CrossEntropyLoss

# ce = CrossEntropyLoss()
# print(ce(x.view(-1, 6), y.view(-1)))
import torch

from lion_pytorch import Lion
from transformers import AutoTokenizer, AutoModel
from config.parser_argument import parserargs
from data_loader_x import PhoAtisDataSet, PhoAtisProcessor
from torch.utils.data import DataLoader
from lstm_crf import LSTM_CRF
import os
import torch
from evaluate import evaluate
tokenizer = AutoTokenizer.from_pretrained(parserargs.bert_type)
proccesor = PhoAtisProcessor(parserargs)

test_dataset = PhoAtisDataSet(tokenizer, proccesor, 'test')
test_dl = DataLoader(
    test_dataset,
    batch_size=parserargs.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=PhoAtisDataSet.pack
)

model = LSTM_CRF(parserargs, proccesor.__lenIntent__(), proccesor.__lenSlot__()).to(parserargs.device)
model.load_state_dict(torch.load(f'./checkpoints/version_1_best.pth'))
test_perf_intent, test_perf_slot, test_perf = evaluate(model, test_dl, parserargs, 'Test', 0)

dev_dataset = PhoAtisDataSet(tokenizer, proccesor, 'dev')
dev_dl = DataLoader(
    dev_dataset,
    batch_size=parserargs.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=PhoAtisDataSet.pack
)
dev_perf_intent, dev_perf_slot, dev_perf = evaluate(model, dev_dl, parserargs, 'Dev', 0)