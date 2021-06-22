import nltk
import re
import pdb

from pycocoevalcap.bleu.bleu import Bleu

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

real_text = './data/' #syn_val_words
test_text = './results/klgan_examples_bsz64_epoch15_v2/013_examplar_gen'

all_sents = []
with open(input_file, 'r')as fin:
    for line in fin:
        #line.decode('utf-8')
        # line = re.sub(r",", "", line)
        # line = clean_str(line)
        # line = line.split()
        all_sents.append(line)

import numpy as np
ans = np.zeros(4)
for i in range(len(all_sents)):
    tmp = all_sents[:]
    pop = tmp.pop(i)
    ref = {0: tmp}
    hop = {0: [pop]}

    ans[3] += score(ref, hop)['Bleu_4']
    ans[2] += score(ref, hop)['Bleu_3']
    ans[1] += score(ref, hop)['Bleu_2']
    ans[0] += score(ref, hop)['Bleu_1']
    # pdb.set_trace()

ans /= len(all_sents)
print('sink: ', ans)


