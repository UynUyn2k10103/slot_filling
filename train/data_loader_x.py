import os
import sys


## Đoạn lệnh để nhảy ra ngoài folder thôi
myDir = os.getcwd()
sys.path.append(myDir)

from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())

sys.path.append(a)


from config.parser_argument import parserargs
import os


class PhoAtisProcessor(object):
    def __init__(self, parserargs):
        self.intent_labels = self.__read_file(os.path.join(parserargs.data_dir, parserargs.type_level, parserargs.intent_label_file))
        self.slot_labels = self.__read_file(os.path.join(parserargs.data_dir, parserargs.type_level, parserargs.slot_label_file))
        self.parserargs = parserargs
    
    
    def __read_file(self, path):
        with open(file = path, mode = 'r', encoding = 'utf8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        return lines
    

    def __lenIntent__(self):
        return len(self.intent_labels)
    

    def __lenSlot__(self):
        return len(self.slot_labels)


    def __getIntent2Index__(self, intent_label):
        label2idx = (self.intent_labels.index(intent_label) if intent_label in self.intent_labels else self.intent_labels.index("UNK"))
        return label2idx
    
    
    def __getSlot2Index__(self, slot_label):
        slot2idx = (self.slot_labels.index(slot_label) if slot_label in self.slot_labels else self.slot_labels.index("UNK"))
        
        return slot2idx


import torch
from torch.utils.data import Dataset
import re
import string
from transformers import AutoTokenizer
import numpy as np
from pytorch_metric_learning import samplers

def keep(items):
    return items

def np_to_tensor(items):
    tensors = [torch.from_numpy(item) for item in items]
    tensors = torch.stack(tensors,dim = 0)
    return tensors


TENSOR_TYPES = {
    'sentences': keep,
    # 'intent_label': torch.LongTensor,
    # 'slot_label': torch.LongTensor,
    'attention_mask': torch.LongTensor,
    'transform_matrix': np_to_tensor,
    'indices': torch.LongTensor,
    'bert_length': torch.LongTensor,
    'word_length': torch.LongTensor,
    'target_intent': torch.LongTensor,
    'target_slots': torch.LongTensor
}

class PhoAtisDataSet(Dataset):
    def __init__(self, tokenizer, processor, type_data):
        # self.processor = PhoAtisProcessor(parserargs)
        
        self.processor = processor
        self.sentences = list(map(self.__format_sentence, self.__get_data(type_data, parserargs.file_sentence)))
        self.labels = self.__get_data(type_data, parserargs.file_intent)
        self.slots = self.__get_data(type_data, parserargs.file_slot)

        self.tokenizer = tokenizer
        self.CLS = self.tokenizer.cls_token_id
        self.PAD = self.tokenizer.pad_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.UNK = self.tokenizer.unk_token_id
        
        # BERT MAX LEN
        self.BML = processor.parserargs.bert_length
        # Word MAX LEN
        self.WML = processor.parserargs.word_length
    
    def __format_sentence(self, sentence):
        # sentence = re.sub(f'[{string.punctuation}\d\n]', ' ', sentence)
        
        tokens = sentence.split()
        sentence = ' '.join(tokens)
        
        return sentence.lower()
    
    
    def __get_data(self, type_data, file):
        path = os.path.join(self.processor.parserargs.data_dir, self.processor.parserargs.type_level, type_data, file)
        
        with open(file = path, mode = 'r', encoding = 'utf8') as f:
            data = [line.strip() for line in f.readlines()]
        
        return data
    
    
    def __len__(self):
        return len(self.sentences)
    
    
    def __getitem__(self, index):
        words = self.sentences[index].split()
        word_length = len(words)
        
        #config labels
        #convert intent label to index
        intent_label = self.labels[index]
        intent2idx = self.processor.__getIntent2Index__(intent_label)
        
        #convert slot label to index
        slots = self.slots[index].split()
        
        slot2idx = [self.processor.__getSlot2Index__(slot) for slot in slots]
        
        assert word_length == len(slot2idx)
        # assert len(slot2idx) == self.WML
    
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
            all_spans.append([start, end])

            if len(pieces) != 0:
                piece_num = len(pieces)
                mean_matrix = np.full((piece_num), 1.0/piece_num)
                transform_matrix[idx+1,start:end] = mean_matrix
                
        all_pieces.append(self.SEP)
        cls_text_sep_length = len(all_pieces)
        transform_matrix[len(words),cls_text_sep_length-1] = 1.0
        assert len(all_pieces) <= self.BML
        
        pad_len = self.BML - len(all_pieces)
        all_pieces += [self.PAD] * pad_len
        attention_mask = [1.0] * cls_text_sep_length + [0.0] * pad_len
        assert len(all_pieces) == self.BML

        #add padding to slot2dix
        pad2idx = [self.processor.__getSlot2Index__("PAD")]
        pad_len = self.BML - len(slot2idx) - 1
        slot2idx =  pad2idx + slot2idx + pad2idx * pad_len
        

        return {
            'sentences': self.sentences[index],
            # 'intent_label': intent_label,
            # 'slot_label': self.slots[index],
            'attention_mask': attention_mask,
            'transform_matrix': transform_matrix,
            'indices': all_pieces,
            'bert_length': cls_text_sep_length,
            'word_length': word_length,
            'target_intent': intent2idx,
            'target_slots': slot2idx,
        }
    
    @staticmethod
    def pack(items):
        # try: 
        return {
            k: TS_TYPE([x[k] for x in items])
            for k, TS_TYPE in TENSOR_TYPES.items()
        }
        # except:
        #     for x in items:
        #         print(len(x['target_slots']))
        #     return {
        #         k: TS_TYPE([x[k] for x in items])
        #         for k, TS_TYPE in TENSOR_TYPES.items()
        #     }
        
            
    def get_sampler(self,m, batch_size= None, length_before_new_iter = 100000):
        labels = self.topics
        sampler = samplers.MPerClassSampler(labels, m, batch_size=batch_size, length_before_new_iter=length_before_new_iter)
        return sampler
    
    def get_hirachical_sampler(self, batch_size, samples_per_class, batches_per_super_tuple =4, super_classes_per_batch = 1):
        labels = [[0,topic] for topic in self.topics]
        labels = np.asarray(labels)
        
        sampler = samplers.HierarchicalSampler(
            labels,
            batch_size = batch_size,
            samples_per_class = samples_per_class,
            batches_per_super_tuple=batches_per_super_tuple,
            super_classes_per_batch=super_classes_per_batch,
            inner_label=0,
            outer_label=1,
        )
        return sampler

if __name__ == '__main__':
    # processor = PhoAtisProcessor(parserargs)
    # print(processor.intent_labels)
    # print(processor.__getIntent2Index__('aircraft'))
    # print(processor.__getIntent2Index__('nooo'))

    # print(processor.slot_labels)
    # print(processor.__getSlot2Index__('I-transport_type'))
    # print(processor.__getSlot2Index__('what'))
    
    tokenizer = AutoTokenizer.from_pretrained(parserargs.bert_type)
    phoAtisDataSet = PhoAtisDataSet(tokenizer, PhoAtisProcessor(parserargs), 'dev')
    
    # phoAtisDataSet.__getItem__(12)
    res = phoAtisDataSet.__getitem__(12)
    print(res)
