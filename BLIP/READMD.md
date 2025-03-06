此文件夹包含 [BLIP 存储库]（https://github.com/salesforce/BLIP）。

使用以下命令安装 BLIP（在上面一个文件夹中）：

```
cd ../
git clone https://github.com/salesforce/BLIP
```



从谷歌驱动下载模型权重

https://drive.google.com/file/d/16HXxAnZzRzTFo9Ay-5nftkRdJCHmUMJA/view



BLIP目录

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

