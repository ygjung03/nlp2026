'''
디코딩 전략 비교 결과를 chrF와 TTR로 비교한다.

사전 조건:
  - decode_compare.py를 실행하여
    predictions/generated_sonnets_decode_{name}.txt 파일들이 생성되어 있어야 한다.

실행:
  python compare_decode.py
'''

import os

from datasets import SonnetsDataset
from evaluation import test_sonnet


def calculate_diversity(text_list):
  """생성된 텍스트들의 어휘 다양성(Type-Token Ratio)을 계산한다."""
  all_words = []
  for text in text_list:
    words = ''.join(c if c.isalnum() else ' ' for c in text).lower().split()
    all_words.extend(words)

  if not all_words:
    return 0.0, 0

  total_tokens = len(all_words)
  unique_types = len(set(all_words))
  ttr = unique_types / total_tokens
  return ttr, unique_types


def main():
  exp_names = ['greedy', 'conservative', 'baseline', 'creative']
  gold_path = 'data/TRUE_sonnets_held_out_dev.txt'

  print(f"{'전략명':<15} | {'temp':>5} | {'top_p':>5} | {'chrF':>8} | {'TTR':>8} | {'고유 단어 수':>10}")
  print('-' * 70)

  temps = {'greedy': 0.1, 'conservative': 0.7, 'baseline': 1.2, 'creative': 1.5}
  top_ps = {'greedy': 1.0, 'conservative': 0.9, 'baseline': 0.9, 'creative': 0.95}

  for name in exp_names:
    test_path = f'predictions/generated_sonnets_decode_{name}.txt'

    if not os.path.exists(test_path):
      print(f"{name:<15} | (결과 파일 없음: {test_path})")
      continue

    chrf_score = test_sonnet(test_path=test_path, gold_path=gold_path)

    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    ttr, unique_words = calculate_diversity(generated_sonnets)

    print(f"{name:<15} | {temps[name]:>5.1f} | {top_ps[name]:>5.02f} | {chrf_score:8.3f} | {ttr:8.2%} | {unique_words:10d}")


if __name__ == "__main__":
  main()