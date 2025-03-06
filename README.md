# 通过查询重写增强的交互式图像检索

## 概述
该项目实现了一个交互式图像检索系统，该系统利用用户反馈结合由大型语言模型（LLMs）和视觉语言模型（VLMs）支持的高级查询重写技术来提高查询质量。该系统在多轮交互环境中运行，允许进行迭代查询优化以提高召回率和 F1 分数，并实现更相关的图像检索。

## 关键特性
-支持多轮交互以进行持续查询优化。
-集成LLM以实现有效的查询去噪。
-利用修改后的MSR-VTT数据集，专为图像检索任务量身定制。
-显着提高召回率，达到最先进的性能。

## 安装

1. **下载 Flickr30k 数据集**  
   将数据集保存到“playground”文件夹中。

   数据集来源：[Kaggle]（https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset）。
   使用下面的bash脚本下载或手动下载数据集。
   下载后的数据集结构如下所示：

   **Playground 结构**:
   ```shell
   playground/
   └── data
       ├── flickr30k
       │   ├── README.md
       │   ├── flickr30k_images
       │   └── images_captions.csv
       ├── mscoco
       │   ├── val2017
       │   ├── README.md
       │   └── images_captions.csv

bash脚本

```bash
Copy the code
#!/bin/bash

# Dataset ID
DATASET_ID="eeshawn/flickr30k"

# Download the dataset
echo "Downloading dataset: $DATASET_ID"
kaggle datasets download -d $DATASET_ID

# Unzip the dataset
ZIP_FILE="${DATASET_ID//\//_}.zip"  # Replace '/' with '_' to generate the zip file name
echo "Unzipping file: $ZIP_FILE"
unzip $ZIP_FILE

# Completion message
echo "Download and extraction complete."

```



2. 克隆BLIP仓库并下载权重

```shell
cd ../
git clone https://github.com/salesforce/BLIP
```



从谷歌驱动下载模型权重

https://drive.google.com/file/d/16HXxAnZzRzTFo9Ay-5nftkRdJCHmUMJA/view

**目录结构**:

```shell
BLIP/
├── BLIP.gif
├── CODEOWNERS
├── CODE_OF_CONDUCT.md
├── LICENSE.txt
├── README.md
├── SECURITY.md
├── chatir_weights.ckpt
├── cog.yaml
├── configs
│   ├── bert_config.json
│   ├── caption_coco.yaml
│   ├── med_config.json
│   ├── nlvr.yaml
│   ├── nocaps.yaml
│   ├── pretrain.yaml
│   ├── retrieval_coco.yaml
│   ├── retrieval_flickr.yaml
│   ├── retrieval_msrvtt.yaml
│   └── vqa.yaml
├── data
│   ├── __init__.py
│   ├── coco_karpathy_dataset.py
│   ├── flickr30k_dataset.py
│   ├── nlvr_dataset.py
│   ├── nocaps_dataset.py
│   ├── pretrain_dataset.py
│   ├── utils.py
│   ├── video_dataset.py
│   └── vqa_dataset.py
├── demo.ipynb
├── eval_nocaps.py
├── eval_retrieval_video.py
├── models
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── blip.cpython-310.pyc
│   │   ├── blip_itm.cpython-310.pyc
│   │   ├── med.cpython-310.pyc
│   │   └── vit.cpython-310.pyc
│   ├── blip.py
│   ├── blip_itm.py
│   ├── blip_nlvr.py
│   ├── blip_pretrain.py
│   ├── blip_retrieval.py
│   ├── blip_vqa.py
│   ├── med.py
│   ├── nlvr_encoder.py
│   └── vit.py
├── predict.py
├── pretrain.py
├── requirements.txt
├── train_caption.py
├── train_nlvr.py
├── train_retrieval.py
├── train_vqa.py
├── transform
│   └── randaugment.py
└── utils.py
```

3. 下载 LLaVA 和 LLaMA 权重

llava

https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

llama3.1

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# 运行

0. **硬件要求**

   - **最低 GPU 要求**: RTX 3090

   ### 依赖

   安装所需的软件包（仅安装下面列出的这些；其他的可以使用最新版本）：

```shell
pip install -U flash-attn --no-build-isolation
pip install transformers timm fairscale pandas faiss-cpu matplotlib bitsandbytes
```

0.2 flash-attention 

如果安装失败，请通过注释掉相关代码来禁用Flash关注：

```python 
   # import flash_attn
   
   self.model = LlavaNextForConditionalGeneration.from_pretrained(
     model_id, 
     torch_dtype=torch.float16, 
     quantization_config=quantization_config if quantization else None,
     # attn_implementation="flash_attention_2",
     low_cpu_mem_usage=True,
     device_map="auto")  
   
   self.model = AutoModelForCausalLM.from_pretrained(
     model_id, 
     torch_dtype=torch.float16, 
     quantization_config=quantization_config if quantization else None,
     # attn_implementation="flash_attention_2",
     device_map="auto")
```

0.3 Quantization

对LLM和多模态模型使用int8量化：

   ```python
   quantization_config = BitsAndBytesConfig(
                 load_in_8bit=True,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.float16,
           )
   ```

如果内存不足，请使用int4量化：

   ```python
   quantization_config = BitsAndBytesConfig(
                 load_in_4bit=True,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.float16,
           )
   ```

如果您有48GB内存，请使用float16精度：

   ```python
   query_edit_model = QueryEditing(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization=False)
   caption_model = Captioning(model_id="llava-hf/llava-v1.6-mistral-7b-hf", quantization=False)
   ```

1. 建立图像索引

   ```python
   python embed_image_as_faiss.py
   ```

2. 在代码中针对 Flickr30k 数据集运行该算法。

 ```python
 python multi_modal_query_expansion.py
 ```

3. 在代码中针对 Mscoco 数据集测试中运行算法

 ```python
 python multi_modal_query_expansion_mscoco.py
 ```
