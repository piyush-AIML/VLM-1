from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(reference, prediction):
    return sentence_bleu([reference.split()], prediction.split())
