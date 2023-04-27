import numpy as np
import torch
from collections import defaultdict
from tqdm.auto import tqdm
from pyemd import emd
from distutils.version import LooseVersion

from transformers import AutoModel
from transformers import logging
logging.set_verbosity_error()

import string

__all__ = []


lang2model = defaultdict(lambda: "microsoft/codebert-base-mlm")
lang2model.update({
        "python": "neulab/codebert-python",
        "javascript": "neulab/codebert-javascript",
        "js": "neulab/codebert-javascript",
        "c": "neulab/codebert-c",
        "cpp": "neulab/codebert-cpp",
        "c++": "neulab/codebert-cpp",
        "java": "neulab/codebert-java",
})

model2layers = defaultdict(lambda: 10)
model2layers.update({
    "microsoft/codebert-base-mlm": 10,
    "neulab/codebert-python": 11,
    "neulab/codebert-javascript": 10,
    "neulab/codebert-c": 10,
    "neulab/codebert-cpp": 10,
    "neulab/codebert-java": 7,
})


def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    else:
        import transformers
        if LooseVersion(transformers.__version__) >= LooseVersion("3.0.0"):
            # return tokenizer.encode(sent, add_special_tokens=True, max_length=tokenizer.model_max_length, truncation=True)
            # Uri: truncation=False
            return tokenizer.encode(sent, add_special_tokens=True, max_length=tokenizer.model_max_length, truncation='do_not_truncate')
        else:
            return tokenizer.encode(sent, add_special_tokens=True, max_length=tokenizer.model_max_length)


def get_model(model_type, num_layers, all_layers=None):
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        else:
            raise ValueError(f"Not supported model architecture: {model_type}")

    return model


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask, tokenizer, all_layers=False, chunk_overlap=0.5):
    model.eval()
    with torch.no_grad():
        model_max_length = tokenizer.model_max_length
        window_indices = get_window_indices(total_seq_length=x.shape[-1], encoder_max_length=model_max_length, chunk_overlap=chunk_overlap)

        chunk_output = []
        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = x[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            out = model(chunk, attention_mask=chunk_attention_mask)
            
            if all_layers:
                emb = torch.stack(out[-1], dim=2)
            else:
                emb = out[0]
            emb = emb[:, update_start_ind:update_end_ind]
            chunk_output.append(emb)

    emb = torch.cat(chunk_output, dim=1)            

    return emb

def get_window_indices(total_seq_length, encoder_max_length, chunk_overlap):
    # Copied from SLED (Ivgy et al., 2022)
    # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
    if total_seq_length <= encoder_max_length:
        return [(0, total_seq_length, 0, total_seq_length)]
    else:
        window_margin = int(encoder_max_length * chunk_overlap / 2)
        results = []
        stride = encoder_max_length - 2 * window_margin
        # if self.chunk_overlap == 0:
        #     stride = self.model_encoder_max_len
        context_start = update_start_ind = 0
        context_end = encoder_max_length
        update_end_ind = context_end - window_margin
        # first window always should update from the beginning
        results.append((context_start, context_end, update_start_ind, update_end_ind))  

        while context_end < total_seq_length:
            context_end = min(total_seq_length, context_end + stride)
            context_start = (
                context_start + stride if context_end < total_seq_length else total_seq_length - encoder_max_length
            )
            update_start_ind = max(update_start_ind + stride, update_end_ind)
            # last window always should update until the end
            update_end_ind = (
                min(total_seq_length, update_end_ind + stride) if context_end < total_seq_length else total_seq_length
            )

            cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                update_end_ind - context_start

            results.append((cs, ce, us, ue))
        return results

def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def collate_idf(arr, tokenizer, idf_dict, device="cuda:0"):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(all_sens, model, tokenizer, idf_dict, batch_size=-1, device="cuda:0", all_layers=False, chunk_overlap=0.5):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens, tokenizer, idf_dict, device=device)

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model, padded_sens[i : i + batch_size], attention_mask=mask[i : i + batch_size], all_layers=all_layers,
                tokenizer=tokenizer,
                chunk_overlap=chunk_overlap,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def mark_all_punc_tokens(tokens):
    whitespace_token='\u0120'
    newline_token = '\u010a'
    tab_token = '\u0109'

    def all_punc(tok):
        return all(c in (set(string.punctuation) - set('+-*/&|~')) for c in tok)

    def strip_whitespace(tok):
        return tok.strip(whitespace_token + newline_token + tab_token)

    return torch.tensor([not all_punc(strip_whitespace(tok)) for tok in tokens])


def get_state_lens(split_sen, tokens, tokenizer):
    state_lens = [1]
    tmp_tokens = []
    for i, sen in enumerate(split_sen):
        if i:
            sen = " " + sen
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(sen)['input_ids'])[1:-1]
        tmp_tokens.extend(tokens)
        state_lens.append(state_lens[-1] + len(tokens))
    # print(tmp_tokens)
    return state_lens


def bert_chunk_cos_score(
    model, refs, hyps, tokenizer, idf_dict, verbose=False, batch_size=64, device="cuda:0", all_layers=False, 
    no_punc=False, sources=None, chunk_overlap=0.5, use_flow=False
):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    precisions, recalls = [], []
    split_sentences = refs + hyps
    refs = [" ".join(r).strip() for r in refs]
    hyps = [" ".join(h).strip() for h in hyps]
    sentences = refs + hyps
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()

    for batch_start in tqdm(iter_range):

        sen_batch = sentences[batch_start : batch_start + batch_size]
        split_sen_batch = split_sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers,
            chunk_overlap=chunk_overlap
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()

        for i, (sen, split_sen) in enumerate(zip(sen_batch, split_sen_batch)):

            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            tokens = tokenizer.convert_ids_to_tokens(tokenizer(sen)['input_ids'])

            # state_lens = get_state_lens(split_sen, tokens, tokenizer)
            state_lens = list(range(1, len(tokens)))

            # The input is of the format: "<bos> <source> <code>"
            # So we the <bos> token and the code, removing the source
            emb_lst = [emb[state_lens[idx]:state_lens[idx+1]] for idx in range(len(state_lens)-1)]
            idf_lst = [idf[state_lens[idx]:state_lens[idx+1]] for idx in range(len(state_lens)-1)]

            token_lst = [tokens[state_lens[idx]:state_lens[idx+1]] for idx in range(len(state_lens)-1)]
            new_emb_lst, new_idf_lst = [], []

            for lidx in range(len(emb_lst)):
                if no_punc:
                    mask = mark_all_punc_tokens(token_lst[lidx])
                    if len(mask) == 0:
                        continue
                    cur_emb = emb_lst[lidx][mask]
                    cur_idf = idf_lst[lidx][mask]
                else:
                    cur_emb = emb_lst[lidx]
                    cur_idf = idf_lst[lidx]
                if 0 in cur_idf.shape:
                    continue
                if cur_idf.sum().cpu().item() == 0:
                    continue
                new_emb_lst.append(cur_emb)
                new_idf_lst.append(cur_idf)
            
            idf_embs = [e * idf.unsqueeze(-1) for e, idf in zip(new_emb_lst, new_idf_lst)]

            # mean reduce embs
            idf_embs = [e.sum(dim=0) / idf.sum() for e, idf in zip(idf_embs, new_idf_lst)]
            idfs = [idf.sum() for idf in new_idf_lst]
            # stack embs_lst
            if len(idf_embs) == 0:
                idf_embs = [torch.ones(768).cpu()]
            idf_embs = torch.stack(idf_embs, dim=0)
            stats_dict[sen] = idf_embs, idfs

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in tqdm(iter_range):
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]
            emb_refs = [stats_dict[r][0] for r in batch_refs]
            emb_hyps = [stats_dict[h][0] for h in batch_hyps]
            idf_refs = [stats_dict[r][1] for r in batch_refs]
            idf_hyps = [stats_dict[h][1] for h in batch_hyps]

            for i, (emb_ref, emb_hyp) in enumerate(zip(emb_refs, emb_hyps)):
                emb_ref.div_(torch.norm(emb_ref, dim=-1).unsqueeze(-1))
                emb_hyp.div_(torch.norm(emb_hyp, dim=-1).unsqueeze(-1))

                if use_flow:
                    # calculate the distance of the two vector matrics
                    dist = torch.cdist(emb_ref.unsqueeze(0), emb_hyp.unsqueeze(0), p=2).squeeze(0)
                    # calculate the flow
                    ref_len, hyp_len = emb_ref.shape[0], emb_hyp.shape[0]
                    ref_weights = np.zeros(ref_len + hyp_len, dtype=np.float64)
                    hyp_weights = np.zeros(ref_len + hyp_len, dtype=np.float64)
                    ref_weights[:ref_len] = 1.0 / ref_len
                    hyp_weights[ref_len:] = 1.0 / hyp_len
                    dist_np = np.zeros((ref_len + hyp_len, ref_len + hyp_len), dtype=np.float64)
                    dist_np[:ref_len, ref_len:] = dist.cpu().numpy()
                    precost = emd(ref_weights, hyp_weights, dist_np)
                    # transpose the distance matrix
                    # dist_np = dist_np.transpose().copy()
                    # recost = emd(hyp_weights, ref_weights, dist_np)
                    precisions.append(-precost + 1)
                    recalls.append(-precost + 1)
                else:
                    cossim = torch.mm(emb_ref, emb_hyp.transpose(0, 1))
                    # distsim = 1 - torch.cdist(emb_ref.unsqueeze(0), emb_hyp.unsqueeze(0), p=2).squeeze(0)
                    stat_precision = cossim.max(dim=0)[0]
                    stat_recall = cossim.max(dim=1)[0]
                    idf_hyp = torch.tensor(idf_hyps[i], dtype=torch.float32, device="cpu").clamp_min_(1e-8)
                    idf_ref = torch.tensor(idf_refs[i], dtype=torch.float32, device="cpu").clamp_min_(1e-8)
                    stat_precision = (stat_precision * idf_hyp).sum() / idf_hyp.sum()
                    stat_recall = (stat_recall * idf_ref).sum() / idf_ref.sum()
                    # stat_precision = cossim.mean(dim=0)
                    # stat_recall = cossim.mean(dim=1)
                    precision = stat_precision.mean().cpu().item()
                    recall = stat_recall.mean().cpu().item()
                    precisions.append(precision)
                    recalls.append(recall)
    return precisions, recalls
