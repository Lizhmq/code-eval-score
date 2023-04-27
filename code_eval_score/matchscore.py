import time
import torch
import numpy as np

from collections import defaultdict
from transformers import AutoTokenizer

from code_eval_score.utils import (
    get_model,
    bert_chunk_cos_score,
    get_bert_embedding,
    lang2model,
    model2layers,
    sent_encode,
)


def calculate(cands, refs, lang, device="cpu", batch_size=1, return_all=False):
    """Calculate MatchScore for a list of candidates and references.
    cands/refs are lists of lists of strings. Each item in cands is a
    list of statements. For example, for a python program with 2
    statements, the input can be:
    cands = [["import numpy as np\n", "np.mean(x)"]]
    If you don't want to match at the statement level, you can also
    pass the whole program as a single string:
    cands = ["import numpy as np\nnp.mean(x)"].
    """
    precision, recall = chunk_score(cands, refs, lang, device=device, batch_size=batch_size)
    if return_all:
        return precision, recall
    return recall

def chunk_score(
    cands,
    refs,
    lang=None,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    return_hash=False,
    no_punc=False,
    sources=None,
    chunk_overlap=0.5,
    use_flow=False
):
    """
    Chunked MatchScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`.
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `no_punc` (bool): Uri: exclude punctuation-only tokens in candidate and reference
        - :param: `sources` (list of str): Uri: a list of a source for each candidate, to be concatenated with the candidates
                   but removed from the similarity computation
        - :param: `chunk_overlap` (float): Uri: how much overlap between chunks, when the input is longer than the models' max length

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have 
                  multiple references, the returned score of this candidate is 
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        print("Only support idf dict or False (not using idf)")
        raise NotImplementedError

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    precision, recall = bert_chunk_cos_score(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
        no_punc=no_punc,
        sources=sources,
        chunk_overlap=chunk_overlap,
        use_flow=use_flow
    )
    end = time.perf_counter()

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    return precision, recall


def plot_example(
    candidate, reference, model_type=None, num_layers=None, lang=None, fname="",
):
    """
    MatchScore metric.

    Args:
        - :param: `candidate` (str): a candidate sentence
        - :param: `reference` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `fname` (str): path to save the output plot
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = get_model(model_type, num_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    idf_dict = defaultdict(lambda: 1.0)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, masks, padded_idf = get_bert_embedding(
        [candidate], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding, masks, padded_idf = get_bert_embedding(
        [reference], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens
    r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
    h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
    sim = sim[1:-1, 1:-1]

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    ax.grid(False)
    plt.xlabel("Reference (tokenized)", fontsize=14)
    plt.ylabel("Candidate (tokenized)", fontsize=14)
    title = "Similarity Matrix"
    plt.title(title, fontsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(
                j,
                i,
                "{:.3f}".format(sim[i, j].item()),
                ha="center",
                va="center",
                color="k" if sim[i, j].item() < 0.5 else "w",
            )

    fig.tight_layout()
    if fname != "":
        plt.savefig(fname, dpi=100)
        print("Saved figure to file: ", fname)
    plt.show()
