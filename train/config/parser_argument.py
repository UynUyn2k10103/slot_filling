import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

parser.add_argument("--data_dir", default="./PhoATIS", type=str, help="The input data dir")
parser.add_argument(
        "--type_level",
        type=str,
        default="syllable-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

parser.add_argument("--file_sentence", default="seq.in", type=str, help="File sentence")
parser.add_argument("--file_intent", default="label", type=str, help="File intent")
parser.add_argument("--file_slot", default="seq.out", type=str, help="File slot")
parser.add_argument("--bert_type", default="vinai/phobert-base", type=str, help="Version of Bert")
parser.add_argument("--bert_length", default= 256, type=int, help="Bert max length")
parser.add_argument("--word_length", default= 100, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--update_bert", default=False, type=bool, help="Update bert?")
parser.add_argument("--use_crf", default=True, type=bool, help="Use CRF?")
parser.add_argument("--hidden_size", default=768, type=int, help="Default hidden size of bert is 768. You can choose any number")
parser.add_argument("--batch_size", default=16, type=int, help="Default batch size is 16. You can choose any number")
parser.add_argument("--device", default = torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str, help="device")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for lion.")
parser.add_argument("--epoch", default=32, type=int, help="The initial epoch/iterator.")

parserargs = parser.parse_args('')



if __name__ == "__main__":
    print(parserargs)