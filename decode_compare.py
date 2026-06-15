'''
디코딩 전략 비교 스크립트.

이미 학습된 baseline 체크포인트를 사용하여,
다양한 temperature / top_p 조합으로 소네트를 생성하고
결과를 predictions/ 에 저장한다.

재학습 없이 생성만 수행하므로 수 분 내에 완료된다.

실행:
  python decode_compare.py --use_gpu
'''

import argparse
import random
import torch
import numpy as np

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT, seed_everything


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default="9_baseline_10-1e-05-sonnet.pt",
                      help="사용할 체크포인트 파일 경로.")
  parser.add_argument("--held_out_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--seed", type=int, default=11711)
  args = parser.parse_args()

  seed_everything(args.seed)
  device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

  # baseline 체크포인트 로드
  print(f"Loading checkpoint from {args.checkpoint}...")
  saved = torch.load(args.checkpoint, weights_only=False)
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out = SonnetsDataset(args.held_out_path)

  # 비교할 디코딩 전략 (이름, temperature, top_p)

  configs = [
      ('greedy',       0.1, 1.0,  0),
      ('conservative', 0.7, 0.9,  0),
      ('baseline',     1.2, 0.9,  0),
      ('creative',     1.5, 0.95, 0),
      ('top_k_50',     1.2, 1.0,  50),
      ('top_k_10',     1.2, 1.0,  10),
  ]

  for name, temp, top_p, top_k in configs:    
    print(f"\n{'='*50}")
    print(f"Generating with: {name} (temperature={temp}, top_p={top_p})")
    print(f"{'='*50}")

    # 각 디코딩 전략마다 동일한 seed로 시작하여 공정한 비교를 보장한다.
    seed_everything(args.seed)

    sonnets = []
    for batch in held_out:
      sonnet_id = batch[0]
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=temp, top_p=top_p, top_k=top_k)[0][0]
      decoded = model.tokenizer.decode(output)
      sonnets.append((sonnet_id, decoded))
      print(f"{decoded}\n")

    # 결과 저장
    out_path = f'predictions/generated_sonnets_decode_{name}.txt'
    with open(out_path, 'w') as f:
      f.write('--Generated Sonnets-- \n\n')
      for sid, text in sonnets:
        f.write(f'\n{sid}\n{text}\n\n')
    print(f"Saved to {out_path}")

  print(f"\n{'='*50}")
  print("All decoding strategies completed.")
  print("Run compare_decode.py to see chrF/TTR comparison.")
  print(f"{'='*50}")


if __name__ == "__main__":
  main()