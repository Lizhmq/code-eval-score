import numpy as np
import torch
import torch.nn as nn
import traceback
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration


SRC_LIST = [
	'This is a very good idea. Although simple, but very insightful.',
	'Can I take a look?',
	'Do not trust him, he is a liar.'
]

TGT_LIST = [
	"That's stupid.",
	"What's the problem?",
	'He is trustworthy.'
]


class CodeT5Scorer:
	def __init__(self, device='cuda:0', lang='python', max_length=2048, checkpoint='Salesforce/codet5-large', load=None):
		self.device = torch.device(device)
		self.max_length = max_length
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
		if load:
			self.model.load_state_dict(torch.load(load, map_location=self.device))
		self.model.eval()
		self.model.to(device)

		# Set up loss
		self.loss_fct = nn.NLLLoss(reduction='none')
		self.lsm = nn.LogSoftmax(dim=1)

	def prob(self, prefixs, gens, tokenizer, batch_size=4):
		""" Score a batch of examples """
		score_list = []
		# for i in tqdm(range(0, len(prefixs), batch_size)):
		for i in range(0, len(prefixs), batch_size):
			prefix_list = prefixs[i: i + batch_size]
			cand_list = gens[i: i + batch_size]
			try:
				with torch.no_grad():
					inputs = tokenizer(prefix_list, text_target=cand_list, padding=True, truncation=True,
					                          max_length=self.max_length, return_tensors="pt").to(self.device)
					labels = inputs["labels"]
					labels[labels == self.tokenizer.pad_token_id] = -100
					cand_lengths = (inputs["labels"] != -100).sum(dim=1)

					outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)

					# print(outputs.loss)
					# print(cand_lengths)
					logits = outputs.logits.view(-1, self.model.config.vocab_size)

					loss = self.loss_fct(self.lsm(logits), labels.view(-1))
					loss = loss.view(labels.shape[0], -1)
					# sqrt_lengths = torch.sqrt(cand_lengths.float())
					# log_lengths = torch.log(cand_lengths.float() + 1)
					loss = loss.sum(dim=1) / cand_lengths
					curr_score_list = [-x.item() for x in loss]
					score_list += curr_score_list

			except RuntimeError:
				traceback.print_exc()
				print(f'source: {prefix_list}')
				print(f'target: {cand_list}')
				exit(0)
		return score_list

	def score(self, srcs, refs, hyps, batch_size):
		assert srcs or refs
		src_scores, ref_scores = None, None
		if srcs:
			src_scores = self.prob(srcs, hyps, self.tokenizer, batch_size)
		if refs:
			ref_scores = self.prob(refs, hyps, self.tokenizer, batch_size)
		if src_scores and ref_scores:
			return [(x + y) / 2 for x, y in zip(src_scores, ref_scores)]
		if src_scores:
			return src_scores
		return ref_scores

	def test(self, batch_size=3):
		""" Test """
		print(self.score([], SRC_LIST, TGT_LIST, batch_size))


def calculate(cands, refs, lang, device, batch_size):
	""" Calculate the GenScore.
	cands/refs are list of programs(string).
	"""
	lang2load = {
		'python': 'Lzzzq/CodeParaphrase-python',
		'java': 'Lzzzq/CodeParaphrase-java',
		'cpp': 'Lzzzq/CodeParaphrase-cpp',
		'javascript': 'Lzzzq/CodeParaphrase-javascript',
		'pyconala': 'Lzzzq/CodeParaphrase-pyconala',
	}
	if lang not in lang2load:
		lang = 'python'
		print(f'Warning: language {lang} not supported, use python instead.')
	scorer = CodeT5Scorer(device=device, checkpoint=lang2load[lang])
	scores = scorer.score(None, refs, cands, batch_size)
	scores = [np.exp(x) for x in scores]
	return scores


if __name__ == '__main__':
	scorer = CodeT5Scorer(device='cpu', checkpoint='Salesforce/codet5-base')
	scorer.test()

