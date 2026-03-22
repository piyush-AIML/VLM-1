from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_smoothing = SmoothingFunction().method1


def compute_bleu(reference, prediction):
    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=_smoothing,
    )
