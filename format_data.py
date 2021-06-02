import glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def one_hot_encode(x_batch, relation_dictionary):
    if isinstance(x_batch, str):
        x_batch = [x_batch]

    one_hot_dict = {'TrIP': 0, 'TrWP': 1, 'TrCP': 2, 'TrAP': 3, 'TrNAP': 4, 'PIP': 5, 'TeRP': 6, 'TeCP': 7}
    y_batch = np.zeros((len(x_batch), 8))
    for i in range(len(x_batch)):
        if i in relation_dictionary:
            for relation in relation_dictionary[i]:
                y_batch[i][one_hot_dict[relation]] = 1

    return y_batch


def format_data():
    all_x_data = []
    all_y_data = []
    for filename in glob.glob("./beth/txt/*.txt"):
        x_data = [x for x in open(filename).readlines()]

        y_data = [x.strip() for x in open('./beth/rel/' + filename.split('\\')[1][:-4] + ".rel").readlines()]
        lines = [int(l.split(':')[-2].split(' ')[1]) for l in y_data]
        relations = [l.split('r=')[1].split('\"')[1] for l in y_data]
        relation_dictionary = {}

        for line, relation in zip(lines, relations):
            if line not in relation_dictionary:
                relation_dictionary[line] = set()
            relation_dictionary[line].add(relation)

        y_data = one_hot_encode(x_data, relation_dictionary)

        for i in range(len(x_data)):
            all_x_data.append(x_data[i])
            all_y_data.append(y_data[i])

    return np.array(all_x_data), np.array(all_y_data)


def encode_data(x_data):

    print("encoding data...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    embeddings_inputs = tokenizer(x_data.tolist())["input_ids"]
    embeddings_inputs = [torch.tensor(x).reshape((1, -1)) for x in embeddings_inputs]

    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    embedded_data = [model.forward(x)['pooler_output'].detach().numpy() for x in embeddings_inputs]
    print("Done!")

    return np.array(embedded_data).reshape(-1, 768)
