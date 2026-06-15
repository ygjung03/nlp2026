'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import math
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# --- 방법 3 (다양성 기반 조기 종료)를 위한 헬퍼 함수 ---
def calculate_diversity(text_list):
  """생성된 텍스트들의 어휘 다양성(Type-Token Ratio)을 계산한다."""
  all_words = []
  for text in text_list:
    # 구두점 제거 및 소문자 변환 후 단어를 분리한다.
    words = ''.join(c if c.isalnum() else ' ' for c in text).lower().split()
    all_words.extend(words)

  if not all_words:
    return 0.0, 0

  total_tokens = len(all_words)
  unique_types = len(set(all_words))
  ttr = unique_types / total_tokens
  return ttr, unique_types

class LoRALinear(nn.Module):
  """LoRA(Low-Rank Adaptation) 래퍼. 원본 Linear 레이어를 감싸서
  저랭크 행렬 A, B만 학습한다. [Hu et al., 2021]"""

  def __init__(self, original_linear, rank=4, alpha=1.0):
    super().__init__()
    self.original = original_linear
    self.rank = rank
    self.alpha = alpha

    in_features = original_linear.in_features
    out_features = original_linear.out_features

    # A는 작은 값으로 초기화, B는 0으로 초기화 → 학습 시작 시 LoRA 출력이 0
    self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
    self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    # 원본 가중치는 동결
    for param in self.original.parameters():
      param.requires_grad = False

  def forward(self, x):
    original_output = self.original(x)
    lora_output = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
    return original_output + lora_output


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # LoRA 사용 시: 원본 가중치를 동결하고 Q, V에 저랭크 어댑터만 학습 [Hu et al., 2021]
    # Full fine-tuning 시: 모든 파라미터를 학습
    if hasattr(args, 'use_lora') and args.use_lora:
      for param in self.gpt.parameters():
        param.requires_grad = False
      for layer in self.gpt.gpt_layers:
        layer.self_attention.query = LoRALinear(layer.self_attention.query, rank=args.lora_rank)
        layer.self_attention.value = LoRALinear(layer.self_attention.value, rank=args.lora_rank)
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    # 1) 모든 위치의 hidden state
    gpt_output = self.gpt(input_ids, attention_mask)
    hidden_states = gpt_output['last_hidden_state']        # [batch, seq_len, hidden_dim]

    # 2) hidden state -> vocab logits
    logits = self.gpt.hidden_state_to_token(hidden_states)  # [batch, seq_len, vocab_size]
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=0, max_length=128):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-k filtering (top-p 이전에 적용하여, 상위 k개 토큰만 남긴다)
      if top_k > 0:
        k = min(top_k, probs.size(-1))
        top_k_probs, top_k_indices = torch.topk(probs, k)
        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
        probs /= probs.sum(dim=-1, keepdim=True)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # 데이터, 해당 데이터셋 및 데이터로더 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  # --- train/val split (방법 1: val_loss 기반 조기 종료에 사용) ---
  # stopping_method가 val_loss가 아니어도 split 자체는 동일하게 적용해서
  # 실험 간 학습 데이터 구성을 동일하게 맞춘다 (공정한 비교를 위해).
  val_size = max(1, int(len(sonnet_dataset) * args.val_split))
  train_size = len(sonnet_dataset) - val_size
  generator = torch.Generator().manual_seed(args.seed)
  train_subset, val_subset = random_split(sonnet_dataset, [train_size, val_size], generator=generator)

  sonnet_dataloader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=sonnet_dataset.collate_fn)
  val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=args.batch_size,
                               collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # --- 조기 종료 관련 상태 변수 ---
  best_val_loss = float('inf')
  patience_counter = 0
  best_ttr = 0.0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
      labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    # --- Perplexity 계산 (방법 2에서 사용, 다른 실험에서도 참고용으로 출력) ---
    try:
      train_ppl = math.exp(train_loss)
    except OverflowError:
      train_ppl = float('inf')

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train perplexity :: {train_ppl :.3f}.")

    # --- Validation Loss 계산 (방법 1에서 사용, 다른 실험에서도 참고용으로 출력) ---
    model.eval()
    val_loss = 0
    num_val_batches = 0
    with torch.no_grad():
      for batch in val_dataloader:
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        val_loss += loss.item()
        num_val_batches += 1
    val_loss = val_loss / num_val_batches
    print(f"Epoch {epoch}: val loss :: {val_loss :.3f}.")

    print('Generating several output sonnets...')
    generated_texts_for_epoch = []
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      generated_texts_for_epoch.append(output[1])
      print(f'{batch[1]}{output[1]}\n\n')

    # --- 어휘 다양성(TTR) 계산 (방법 3에서 사용, 다른 실험에서도 참고용으로 출력) ---
    current_ttr, unique_words_count = calculate_diversity(generated_texts_for_epoch)
    print(f"Epoch {epoch}: Vocabulary Diversity (TTR) :: {current_ttr:.2%} | Unique Words :: {unique_words_count}")

    # 매 epoch 체크포인트는 항상 저장 (baseline 비교 및 디버깅용).
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

    # 최소 학습 epoch을 채우기 전에는 조기 종료를 판단하지 않는다.
    if epoch + 1 < args.min_epochs:
      continue

    # --- 조기 종료 / best 체크포인트 선택 로직 ---
    if args.stopping_method == 'val_loss':
      # 방법 1: 검증 손실이 개선되면 best로 저장하고, patience 동안 개선이 없으면 종료.
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model(model, optimizer, args, f'best_{args.filepath}')
      else:
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter} out of {args.patience}")
        if patience_counter >= args.patience:
          print(f"Early stopping triggered at epoch {epoch}. Training stopped.")
          break

    elif args.stopping_method == 'perplexity':
      # 방법 2: train perplexity가 목표치 이하로 떨어지면 그 시점 모델을 best로 저장하고 종료.
      if train_ppl <= args.target_perplexity:
        print(f"Target perplexity ({args.target_perplexity}) reached! "
              f"Stopping early to prevent over-memorization.")
        save_model(model, optimizer, args, f'best_{args.filepath}')
        break

    elif args.stopping_method == 'diversity':
      # 방법 3: TTR이 개선되면 best로 저장. TTR이 임계값 아래로 떨어지면 모드 붕괴로 보고 종료.
      if current_ttr > best_ttr:
        best_ttr = current_ttr
        save_model(model, optimizer, args, f'best_{args.filepath}')

      if current_ttr < args.min_ttr:
        print(f"WARNING: Model is suffering from Mode Collapse. "
              f"Diversity dropped below {args.min_ttr:.2%}")
        print(f"Early stopping triggered at epoch {epoch}.")
        break

    # args.stopping_method == 'none' (baseline)인 경우 추가 동작 없이 다음 epoch으로.

  return epoch  # 실제로 학습이 끝난 마지막 epoch 번호를 반환 (체크포인트 로딩에 사용).


@torch.no_grad()
def generate_submission_sonnets(args, last_epoch):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # 모든 실험에서 동일한 규칙으로 체크포인트를 선택한다:
  # best_{filepath}가 있으면 best를, 없으면 학습이 끝난 마지막 epoch의 체크포인트를 사용한다.
  best_path = f'best_{args.filepath}'
  if os.path.exists(best_path):
    print(f"Loading the best checkpoint from {best_path}...")
    saved = torch.load(best_path, weights_only=False)
  else:
    last_path = f'{last_epoch}_{args.filepath}'
    print(f"No best checkpoint found. Loading the last-epoch checkpoint from {last_path}...")
    saved = torch.load(last_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Top-k sampling parameter. 0이면 비활성화.", default=0)
  
  # LoRA 관련
  parser.add_argument("--use_lora", action='store_true', help="LoRA 적용 여부.")
  parser.add_argument("--lora_rank", type=int, default=4, help="LoRA의 rank (저랭크 차원).")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # --- 실험 비교를 위한 인자들 ---
  parser.add_argument("--exp_name", type=str, default="baseline",
                      help="실험 이름. 결과/체크포인트 파일명을 구분하는 데 사용된다.")
  parser.add_argument("--stopping_method", type=str,
                      choices=["none", "val_loss", "perplexity", "diversity"],
                      default="none",
                      help="조기 종료 방법: none(=baseline), val_loss, perplexity, diversity 중 선택.")
  parser.add_argument("--val_split", type=float, default=0.1,
                      help="train set 중 validation으로 떼어낼 비율.")
  parser.add_argument("--min_epochs", type=int, default=3,
                      help="조기 종료를 판단하기 시작하기 전 최소 학습 epoch 수.")

  # 방법 1: validation loss 기반
  parser.add_argument("--patience", type=int, default=2,
                      help="val_loss 방식에서 개선이 없을 때 몇 epoch까지 기다릴지.")

  # 방법 2: perplexity threshold 기반
  parser.add_argument("--target_perplexity", type=float, default=5.0,
                      help="perplexity 방식에서 학습을 멈출 목표 perplexity.")

  # 방법 3: 생성 다양성(TTR) 기반
  parser.add_argument("--min_ttr", type=float, default=0.4,
                      help="diversity 방식에서 mode collapse로 판단할 TTR 임계값.")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # 실험 이름을 체크포인트/결과 파일명에 포함시켜 실험별로 구분되게 한다.
  args.filepath = f'{args.exp_name}_{args.epochs}-{args.lr}-sonnet.pt'
  args.sonnet_out = f'predictions/generated_sonnets_{args.exp_name}.txt'

  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  last_epoch = train(args)
  generate_submission_sonnets(args, last_epoch)