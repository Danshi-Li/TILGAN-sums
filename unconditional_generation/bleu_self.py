import nltk
import re
import pdb
import numpy as np
import time

from pycocoevalcap.bleu.bleu import Bleu
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(5), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "Bleu_5"])
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

def bleu_self(input_file):
    # input_file = './results/klgan_examples_bsz64_epoch15_v2/013_examplar_gen' #syn_val_words
    all_sents = []
    with open(input_file, 'r')as fin:
        for line in fin:
            #line.decode('utf-8')
            # line = re.sub(r",", "", line)
            # line = clean_str(line)
            # line = line.split()
            all_sents.append(line)

    ans = np.zeros(5)
    for i in range(len(all_sents)):
        tmp = all_sents[:]
        pop = tmp.pop(i)
        ref = {0: tmp}
        hop = {0: [pop]}

        # ans[3] += score(ref, hop)['Bleu_4']
        # ans[2] += score(ref, hop)['Bleu_3']
        # ans[1] += score(ref, hop)['Bleu_2']
        # ans[0] += score(ref, hop)['Bleu_1']

        bleu_score = score(ref, hop)
        ans[4] += bleu_score['Bleu_5']
        ans[3] += bleu_score['Bleu_4']
        ans[2] += bleu_score['Bleu_3']
        ans[1] += bleu_score['Bleu_2']
        ans[0] += bleu_score['Bleu_1']
        # pdb.set_trace()

    ans /= len(all_sents)
    # print('sink: ', ans)
    print("bleu_self: ", ans)
    return ans

if __name__ == "__main__":
    start = time.time()
    # ans = bleu_self('./results/klgan_examples_bsz64_epoch15_v2/013_examplar_gen')
    ans = bleu_self("./results/newsdata_klgan2020-05-10-09-11-24/008_examplar_gen")
    print("elapsed time = ", time.time() - start)
