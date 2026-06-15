'''
여러 실험(stopping_method)의 결과를 chrF와 TTR(어휘 다양성) 두 지표로 비교한다.

사전 조건:
  - sonnet_generation.py를 exp_name별로 실행해서
    predictions/generated_sonnets_{exp_name}.txt 파일들이 이미 생성되어 있어야 한다.
  - data/TRUE_sonnets_held_out_dev.txt (정답 14줄 전체)가 존재해야 한다.

실행:
  python compare_experiments.py
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
  # 비교할 실험 이름들. sonnet_generation.py 실행 시 사용한 --exp_name과 일치해야 한다.
  exp_names = ['baseline', 'val_loss', 'ppl', 'ttr']

  gold_path = 'data/TRUE_sonnets_held_out_dev.txt'

  print(f"{'실험명':<12} | {'chrF':>8} | {'TTR':>8} | {'고유 단어 수':>10}")
  print('-' * 50)

  for exp_name in exp_names:
    test_path = f'predictions/generated_sonnets_{exp_name}.txt'

    if not os.path.exists(test_path):
      print(f"{exp_name:<12} | (결과 파일 없음: {test_path})")
      continue

    # --- chrF 계산 ---
    chrf_score = test_sonnet(test_path=test_path, gold_path=gold_path)

    # --- TTR 계산 ---
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    ttr, unique_words = calculate_diversity(generated_sonnets)

    print(f"{exp_name:<12} | {chrf_score:8.3f} | {ttr:8.2%} | {unique_words:10d}")


if __name__ == "__main__":
  main()