from sklearn.metrics import precision_score, recall_score, f1_score
import tqdm

def metrics(all_golds, all_preds, labels =  None):
    p = precision_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    r = recall_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    f = f1_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    
    return {'p': p * 100, 'r': r * 100, 'f': f * 100}
    


def evaluate(model, dl, args, msg='Test', global_iter = 0):
    model.eval()
    all_golds_intent = []
    all_golds_slot = []
    all_preds_intent = []
    all_preds_slot = []

    for batch in tqdm.tqdm(dl, desc=msg):
        golds_intent = batch['target_intent'].numpy().tolist()
        all_golds_intent += golds_intent

        golds_mul_slot = batch['target_slots'].numpy().tolist()
        for golds_slot in golds_mul_slot:
            all_golds_slot += golds_slot


        logits_intent, preds_intent, logits_slot, preds_mul_slot, loss_slot = model(batch)

        
        preds_intent = preds_intent.cpu().numpy().tolist()
        all_preds_intent += preds_intent
        
        preds_mul_slot = preds_mul_slot.cpu().numpy().tolist()
        for preds_slot in preds_mul_slot:
            all_preds_slot += preds_slot
        

    perfs_intent = metrics(all_golds_intent, all_preds_intent, labels = None)
    pers_slot = metrics(all_golds_slot, all_preds_slot, labels = None)
    
    prefs = {
        'p': (perfs_intent['p'] + pers_slot['p']) / 2, 
        'r': (perfs_intent['r'] + pers_slot['r']) / 2,
        'f': (perfs_intent['f'] + pers_slot['f']) / 2,
    }
    print('intent:')
    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                         perfs_intent['p'],
                                         perfs_intent['r'],
                                         perfs_intent['f'],
                                         ))
    print('slot:')
    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                         pers_slot['p'],
                                         pers_slot['r'],
                                         pers_slot['f'],
                                         ))
    print('total:')
    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                         prefs['p'],
                                         prefs['r'],
                                         prefs['f'],
                                         ))
    return perfs_intent, pers_slot, prefs