import numpy as np
import torch
import random

def fix_randomness(seed:int):
    random.seed(seed) # 데이터 batch 시드 고정
    np.random.seed(seed) # numpy 시드 고정
    torch.manual_seed(seed) # torch 시드 고정
    torch.backends.cudnn.deterministic = True # 딥러닝 프레임워크 사용시 랜덤성 고정 - 다만 사용 시 연산처리 속도가 감소할 수 있음
    torch.backends.cudnn.benchmark = False # 딥러닝 프레임워크 사용시 랜덤성 고정
    torch.cuda.manual_seed(seed) # 모델의 학습 weight 달라질 시 고정
    # torch.cuda.manual_seed_all(random_seed) 멀티 gpu 사용 시