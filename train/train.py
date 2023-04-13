from lion_pytorch import Lion
from transformers import AutoTokenizer, AutoModel
from config.parser_argument import parserargs
from data_loader_x import PhoAtisDataSet, PhoAtisProcessor
from torch.utils.data import DataLoader
from lstm_crf import LSTM_CRF
from torch.nn import CrossEntropyLoss
import tqdm
import os
import torch
from evaluate import evaluate

from comet_ml import Experiment

experiment = Experiment(
    api_key ='fkybbwdQFJtEkcypdS2awGdOz',
    project_name = 'slot-filling',
    workspace = 'uynuyn2k10103'
)
experiment.log_parameters(parserargs)

def save_model(model, path, name):
    os.makedirs(os.path.join(path),exist_ok = True)
    # print(os.path.join(path, name))
    torch.save(model.state_dict(), os.path.join(path, name))


tokenizer = AutoTokenizer.from_pretrained(parserargs.bert_type)
proccesor = PhoAtisProcessor(parserargs)

train_dataset = PhoAtisDataSet(tokenizer, proccesor, 'train')
dev_dataset = PhoAtisDataSet(tokenizer, proccesor, 'dev')
test_dataset = PhoAtisDataSet(tokenizer, proccesor, 'test')

train_dl = DataLoader(
    train_dataset,
    batch_size=parserargs.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=PhoAtisDataSet.pack
)

test_dl = DataLoader(
    test_dataset,
    batch_size=parserargs.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=PhoAtisDataSet.pack
)

dev_dl = DataLoader(
    dev_dataset,
    batch_size=parserargs.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=PhoAtisDataSet.pack
)



model = LSTM_CRF(parserargs, proccesor.__lenIntent__(), proccesor.__lenSlot__()).to(parserargs.device)

params = [x for x in model.parameters() if x.requires_grad]

optimizer = Lion(params, lr=parserargs.learning_rate)
ce = CrossEntropyLoss()



global_iter = 0
best_dev = {'r': 0, 'p': 0, 'f': 0} 

for epoch in range(parserargs.epoch):
        model.train()
        bar = tqdm.tqdm(train_dl, desc='Training', total=len(train_dl))
        for batch in bar:
            
            global_iter += 1
            logits_intent, preds_intent, logits_slot, preds_slot, loss_slot = model(batch)
            
            # print(preds_slot.shape)
            # print(batch['target_slots'].size())
            
            
            # print(logits_intent.shape, logits_slot.shape)
            # print(batch['target_intent'].size())
            

            loss_intent = ce(logits_intent, batch['target_intent'].to(parserargs.device))
            loss = loss_intent + loss_slot / 50 # tìm max, min của từng loại loss rồi normal tính tổng loss có vẻ fair hơn
            

            if global_iter % 10 == 0:
                l = loss.detach().cpu().numpy()
                experiment.log_metric("Train_loss", l, step = global_iter, epoch = epoch)
                
                bar.set_description(f'Training: Loss={l:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        dev_perf_intent, dev_perf_slot, dev_perf = evaluate(model, dev_dl, parserargs, 'Dev', global_iter)
        
        experiment.log_metric("F1_score_dev", dev_perf['f'], step = global_iter, epoch = epoch)
        
        if dev_perf['f'] > best_dev['f']:
            best_dev = dev_perf
            print('New best @ {}'.format(epoch))
            save_model(model, 'checkpoints', f'version_1_best.pth')

# Evaluation test:
model.load_state_dict(torch.load(f'./checkpoints/version_1_best.pth'))
test_perf_intent, test_perf_slot, test_perf = evaluate(model, test_dl, parserargs, 'Test', 0)

with open('./checkpoints/test.txt', mode= 'w+') as f:
    
    f.write('test_perf_intent\n')

    f.write('r:'+  str(test_perf_intent['r']) + '\n')
    f.write('p:'+  str(test_perf_intent['p']) + '\n')
    f.write('f:'+  str(test_perf_intent['f']) + '\n')
    f.write('\n\n')

    f.write('test_perf_slot\n')

    f.write('r:'+  str(test_perf_slot['r']) + '\n')
    f.write('p:'+  str(test_perf_slot['p']) + '\n')
    f.write('f:'+  str(test_perf_slot['f']) + '\n')
    f.write('\n\n')

    f.write('test_perf\n')
    
    f.write('r:'+  str(test_perf['r']) + '\n')
    f.write('p:'+  str(test_perf['p']) + '\n')
    f.write('f:'+  str(test_perf['f']) + '\n')
    f.write('\n\n')
    
     
experiment.log_metric("F1_score_test", test_perf['f'])