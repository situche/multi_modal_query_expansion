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
