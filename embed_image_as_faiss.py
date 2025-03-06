import os
import pickle
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm 
# !pip install timm fairscale
class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """ model projects image to vector, processor load and prepare image to the model"""
        self.model = model
        self.processor = preprocessor


def BLIP_BASELINE():
    from BLIP.models.blip_itm import blip_itm

    model = blip_itm(pretrained='BLIP/chatir_weights.ckpt',  
                     med_config='BLIP/configs/med_config.json',
                     image_size=224,
                     vit='base')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    def blip_project_img(image):
        embeds = model.visual_encoder(image)
        projection = model.vision_proj(embeds[:, 0, :])
        return F.normalize(projection, dim=-1)

    def blip_prep_image(path):
        raw = Image.open(path).convert('RGB')
        return transform_test(raw)

    image_embedder = ImageEmbedder(blip_project_img, blip_prep_image)

    return model, image_embedder


def encode_images_to_faiss(image_paths, index_path='faiss.index', id2image_path_path='id2image_path.pkl', image_embedding_path='image_embedding.pkl'):
    model, image_embedder = BLIP_BASELINE()
    
    # Prepare Faiss index
    dim = 256  # BLIP嵌入大小（根据您的型号进行调整）
    index = faiss.IndexFlatIP(dim)  # 内积（当向量被归一化时为余弦相似度）

    all_embeddings = []
    id2image_path = {}

    for idx, image_path in tqdm(enumerate(image_paths)):
        image = image_embedder.processor(image_path).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        embedding = image_embedder.model(image).cpu().detach().numpy()
        all_embeddings.append(embedding)
        id2image_path[idx] = image_path

    all_embeddings = np.vstack(all_embeddings)  # 堆叠所有嵌入
    index.add(all_embeddings)  # 将嵌入添加到索引

    # Save the index and mappings
    faiss.write_index(index, index_path)
    with open(id2image_path_path, 'wb') as f:
        pickle.dump(id2image_path, f)
    with open(image_embedding_path, 'wb') as f:
        pickle.dump(all_embeddings, f)

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved id2image path mapping to {id2image_path_path}")
    print(f"Saved image embeddings to {image_embedding_path}")

if __name__ == '__main__':
    # 相同的代码在不同的数据集上运行
    # 加载图片路径
    image_fold = './playground/data/flickr30k/flickr30k_images'
    image_paths = os.listdir(image_fold)#[:50]
    merge_image_path = [os.path.join(image_fold, p) for p in image_paths]
    image_paths = sorted(merge_image_path)
    
    # id2image = dict(zip(range(len(merge_image_path)), merge_image_path))
    
    # Encode images and create FAISS index
    encode_images_to_faiss(image_paths,
                          index_path='./checkpoints/m_query_expansion_faiss.index', 
                           id2image_path_path='./checkpoints/m_query_expansion_id2image_path.pkl', 
                           image_embedding_path='./checkpoints/m_query_expansion_image_embedding.pkl')

    # # Find similar images for a given text
    # text_query = "A description of the image you're looking for."
    # similar_images, distances = find_similar_images(text_query, k=5)
    # print("Top similar images:", similar_images)
    # print("Distances:", distances)

    image_fold = './playground/data/mscoco/val2017'
    image_paths = os.listdir(image_fold)  # [:50]
    merge_image_path = [os.path.join(image_fold, p) for p in image_paths]
    image_paths = sorted(merge_image_path)

    # id2image = dict(zip(range(len(merge_image_path)), merge_image_path))

    # Encode images and create FAISS index
    encode_images_to_faiss(image_paths,
                           index_path='./checkpoints/m_query_expansion_faiss_mscoco.index',
                           id2image_path_path='./checkpoints/m_query_expansion_id2image_path_mscoco.pkl',
                           image_embedding_path='./checkpoints/m_query_expansion_image_embedding_mscoco.pkl')
