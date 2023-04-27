from code_eval_score import matchscore
from code_eval_score import genscore


s1 = """
x ** 0.5
"""

s2 = """
math.sqrt(x)
"""

matchscore.calculate(
    cands=[[s1]],
    refs=[[s1]],
    lang="python",
    device="cpu",
    batch_size=1,
)

res = genscore.calculate(
    cands=[s1],
    refs=[s1],
    lang="python",
    device="cpu",
    batch_size=1,
)
print(res)