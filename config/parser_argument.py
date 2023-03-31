import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

parser.add_argument("--data_dir", default="/kaggle/input/phoatis/PhoATIS", type=str, help="The input data dir")
parser.add_argument(
        "--type_level",
        type=str,
        default="syllable-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

parser.add_argument("--file_sentence", default="seq.in", type=str, help="File sentence")
parser.add_argument("--file_intent", default="label", type=str, help="File intent")
parser.add_argument("--file_slot", default="seq.out", type=str, help="File slot")

parserargs = parser.parse_args('')

print(parserargs)