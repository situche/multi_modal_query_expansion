playground的结构

```shell
playground/
└── data
    ├── flickr30k
    │   ├── __MACOSX
    │   │   └── flickr30k-images
    │   └── flickr30k_images

```



flickr30k_images

使用kaggle下载：https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

要下载 Kaggle 数据集 `flickr30k`，可以使用 Kaggle API。下面是具体步骤，包括如何设置和下载该数据集。

### 步骤 1: 安装 Kaggle API

如果你还没有安装 Kaggle API，请运行以下命令：

```
bash


复制代码
pip install kaggle
```

### 步骤 2: 获取 Kaggle API Token

1. **登录到 Kaggle**: 访问 [Kaggle](https://www.kaggle.com/) 并登录你的账号。
2. **访问你的账户设置**:
   - 点击右上角的头像，然后选择 "My Account"。
3. **下载 API Token**:
   - 在 "API" 部分，点击 "Create New API Token"。这将下载一个 `kaggle.json` 文件到你的计算机。

### 步骤 3: 配置 Kaggle API

1. 将 `kaggle.json` 文件移动到 `~/.kaggle/` 目录（如果这个目录不存在，请创建它）：

   ```
   bash
   
   
   复制代码
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/
   ```

2. 设置 `kaggle.json` 文件的权限，以确保只有你自己可以访问：

   ```
   bash
   
   
   复制代码
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 步骤 4: 下载 `flickr30k` 数据集

使用以下命令下载 `flickr30k` 数据集：

```
bash


复制代码
kaggle datasets download -d eeshawn/flickr30k
```

### 步骤 5: 解压数据集

下载完成后，数据集通常会是一个 zip 文件，你需要解压它：

```
bash


复制代码
unzip flickr30k.zip
```

### 完整的 Shell 脚本示例

你也可以创建一个 Shell 脚本来自动化这个过程。以下是一个示例脚本：

```
bash


复制代码
#!/bin/bash

# 数据集 ID
DATASET_ID="eeshawn/flickr30k"

# 下载数据集
echo "Downloading dataset: $DATASET_ID"
kaggle datasets download -d $DATASET_ID

# 解压数据集
ZIP_FILE="${DATASET_ID//\//_}.zip"  # 将 '/' 替换为 '_' 以生成 zip 文件名
echo "Unzipping file: $ZIP_FILE"
unzip $ZIP_FILE

# 完成
echo "Download and extraction complete."
```

### 使用脚本

1. 将上述脚本保存到一个文件，例如 `download_flickr30k.sh`。

2. 给脚本添加执行权限：

   ```
   bash
   
   
   复制代码
   chmod +x download_flickr30k.sh
   ```

3. 运行脚本：

   ```
   bash
   
   
   复制代码
   ./download_flickr30k.sh
   ```

### 注意事项

- 确保你已安装 `kaggle` CLI 工具，并且能在命令行中运行 `kaggle` 命令。
- 该脚本假定你已将 `kaggle.json` 文件放置在 `~/.kaggle/` 目录中，并已正确配置。
