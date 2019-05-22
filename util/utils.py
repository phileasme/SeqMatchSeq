import os
import torch


def MAP(ground_label: torch.FloatTensor, predict_label: torch.FloatTensor):
    map = 0
    map_idx = 0
    extracted = {}

    for idx_, glab in enumerate(ground_label):
        if ground_label[idx_] != 0:
            extracted[idx_] = 1

    val, key = torch.sort(predict_label, 0, True)
    for i, idx_ in enumerate(key):
        if idx_.tolist() in extracted:
            map_idx += 1
            map += map_idx / (i + 1)

    assert (map_idx != 0)
    map = map / map_idx
    return map


def MRR(ground_label: torch.FloatTensor, predict_label: torch.FloatTensor):
    mrr = 0
    map_idx = 0
    extracted = {}

    for idx_, glab in enumerate(ground_label):
        if ground_label[idx_] != 0:
            extracted[idx_] = 1

    val, key = torch.sort(predict_label, 0, True)
    for i, idx_ in enumerate(key):
        if idx_.tolist() in extracted:
            mrr = 1.0 / (i + 1)
            break

    assert (mrr != 0)
    return mrr


def save(path, config, result, epoch):
    assert os.path.isdir(path)
    recPath = path + config.task + str(config.expIdx) + 'Record.txt'

    file = open(recPath, 'a')
    if epoch == 0:
        for name, val in vars(config).items():
            file.write(name + '\t' + str(val) + '\n')

    file.write(config.task + ': ' + str(epoch) + ': ')
    for i, vals in enumerate(result):
        for _, val in enumerate(vals):
            file.write('%s, ' % val)

        if i == 0:
            print("Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
        elif i == 1:
            print("Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))
        else:
            print("Train: MAP: %s, MRR: %s" % (vals[0], vals[1]))

    file.write('\n')
    file.close()
