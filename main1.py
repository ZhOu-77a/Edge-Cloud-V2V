# import torch
# import time
# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# def occupy(device, gb=23):

#     num_elements = gb * 1024**3 // 4  # float32: 4 bytes
#     x = torch.empty(num_elements, dtype=torch.float32, device=device)
#     return x

# if __name__ == "__main__":
#     tensors = []
#     for i in range(4):
#         device = torch.device(f"cuda:{i}")
#         print(f"Occupying GPU {i} ...")
#         t = occupy(device, gb=23)
#         tensors.append(t)

#     while True:
#         time.sleep(60)

# nohup python -u main.py > output.log 2>&1 &

import torch
import time
import os

# 1. 限制程序只能看到物理显卡 3
# 这样不仅代码写起来简单，还能防止误伤其他显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def occupy(device, gb=3):
    # 计算 float32 元素数量 (1 float = 4 bytes)
    num_elements = gb * 1024**3 // 4 
    print(f"Allocating {gb}GB on {device}...")
    x = torch.empty(num_elements, dtype=torch.float32, device=device)
    return x

if __name__ == "__main__":
    tensors = []
    
    # 2. 这里必须写 "cuda:0"
    # 因为上面屏蔽了其他卡，PyTorch 只能看到一张卡，且索引重置为 0
    # 此时的 cuda:0 就等于 物理显卡 3
    device = torch.device("cuda:0")
    
    try:
        print(f"Starting occupation on Physical GPU 3...")
        t = occupy(device, gb=23) # 根据你的显存大小调整 gb 数值
        tensors.append(t)
        print("Occupation success. Sleeping...")
        
        while True:
            time.sleep(60)
            
    except RuntimeError as e:
        print(f"Error: 显存不足或设备错误 - {e}")

#  nohup python -u main1.py > output.log 2>&1 &
#  nohup python -u wan2.2_v2v_pipeline_inter_round.py.py > wan2.2_v2v_pipeline_inter_round.py.log 2>&1 &