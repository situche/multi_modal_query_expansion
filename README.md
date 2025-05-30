# 查询扩写与图像检索项目说明文档

## 1. 数据集描述

### 1.1. **数据集来源**  
   数据集使用公共数据集 Flickr30k（https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset） 与 MS COCO（https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset）

### 1.2 数据格式
Flickr30k
每条数据包含：
```json
{
    "image_id": "1234567.jpg",
    "captions": [
        "A man in a blue shirt is climbing a rock wall",
        "Someone is mountain climbing on an indoor wall",
        "A person in blue is climbing an indoor rock wall",
        "A man wearing blue is rock climbing indoors",
        "The man scales the indoor climbing wall"
    ]
}
```
MS COCO
每条数据包含：
```json
{
    "info": {
        "description": "COCO 2017 Dataset",
        "year": 2017,
        "version": "1.0",
        "contributor": "COCO Consortium"
    },
    // 其他数据
}
```

## 2. 模型选型
| 完整名称 | 参数量 | 架构类型 |
|----------|--------|----------|
|BLIP| 4.46亿 | Encoder-Decode |
|llava-v1.6-mistral-7b-hf| 70亿 | Decoder-Only |
|Llama-3.1-8B-Instruct| 80亿 | Decoder-Only |

---

## 3. 环境依赖

### 3.1 核心依赖包
```python
Python == 3.12.2
torch == 2.5.1
numpy == 1.26.4
transformers == 4.49.0
faiss == 1.9.0
tqdm == 4.66.5
torchvision == 0.20.1
matplotlib == 3.9.2
```

### 3.2 环境配置
#### 使用conda创建虚拟环境
```python
conda create -n re python=3.12 -y
conda activate re
```
---

## 4. 项目结构与运行
### 4.1 项目文件结构
```text
RE/
├── config.py        # 参数解析模块
├── Qwen_SFT.py      # 主训练脚本
├── Dpsk_SFT.py
├── roberta&ModernBERT.py 
├── train.jsonl      # 数据集
└── test.jsonl
```

### 4.2 Shell运行命令集
```bash
python Qwen_SFT.py \
  --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_data "./data/train.jsonl" \
  --output_dir "./output" \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 3e-5 \
  --warmup_ratio 0.1 \
  --其他需要添加的参数
```
