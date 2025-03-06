# Interactive Image Retrieval Enhanced by Query Rewriting

## Overview
This project implements an interactive image retrieval system that enhances query quality using user feedback combined with advanced query rewriting techniques powered by large language models (LLMs) and vision-language models (VLMs). The system operates in a multi-round interaction environment, allowing iterative query optimization to improve recall rates  and f1 score and achieve more relevant image retrieval.

## Key Features
- Supports multi-round interactions for continuous query optimization.
- Integrates LLMs for effective query denoising.
- Utilizes a modified MSR-VTT dataset, specifically tailored for image retrieval tasks.
- Achieves significant improvements in recall rates, reaching state-of-the-art performance.

## Installation

1. **Download the Flickr30k Dataset**  
   Save the dataset into the `playground` folder.  

   Dataset source: [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).  
   Use the bash script below to download or manually download the dataset.  
   The dataset structure after download will look like this:

   **Playground Structure**:
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



2. Clone the BLIP Repository and Download the Weights

```shell
cd ../
git clone https://github.com/salesforce/BLIP
```



Download model weight from google driver

https://drive.google.com/file/d/16HXxAnZzRzTFo9Ay-5nftkRdJCHmUMJA/view

**Directory Structure**:

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

3. Download LLaVA and LLaMA Weights

llava

https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

llama3.1

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Execution

0. **Hardware Requirements**

   - **Minimum GPU**: RTX 3090

   ### Dependencies

   Install the required packages (only the ones listed below; others can use the latest versions):

```shell
pip install -U flash-attn --no-build-isolation
pip install transformers timm fairscale pandas faiss-cpu matplotlib bitsandbytes
```

0.2 flash-attention 

If installation fails, disable Flash Attention by commenting out the relevant code:

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

Use int8 quantization for LLMs and multi-modal models:

   ```python
   quantization_config = BitsAndBytesConfig(
                 load_in_8bit=True,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.float16,
           )
   ```

If memory is insufficient, use int4 quantization:

   ```python
   quantization_config = BitsAndBytesConfig(
                 load_in_4bit=True,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.float16,
           )
   ```

If you have 48GB memory, use float16 precision:

   ```python
   query_edit_model = QueryEditing(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization=False)
   caption_model = Captioning(model_id="llava-hf/llava-v1.6-mistral-7b-hf", quantization=False)
   ```

1. Build the image index

   ```python
   python embed_image_as_faiss.py
   ```

2. Run the algorithm in Jupyter Notebook for Flickr30k benchmark

 ```
 multi_modal_query_expansion.ipynb
 ```

3. Run the algorithm in Jupyter Notebook for Mscoco benchmark

 ```
 multi_modal_query_expansion_mscoco.ipynb
 ```