import torch
import faiss
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import (LlavaNextProcessor, LlavaNextForConditionalGeneration, 
                          AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig,
                          )

from PIL import Image
import json 
import random
import flash_attn
import matplotlib.pyplot as plt
import os
import numpy as np 
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def show_image(image_paths, sentence=None):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axs = axs.flatten()

    for ax, img_path in zip(axs, image_paths):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(img_path.split('/')[-1])
        except FileNotFoundError:
            ax.imshow(np.zeros((10, 10, 3), dtype=int))
            ax.axis('off')
            ax.set_title('File Not Found')
    if sentence:
        fig.suptitle(sentence, fontsize=16)
    plt.tight_layout()
    plt.show()

class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """ model projects image to vector, processor load and prepare image to the model"""
        self.model = model
        self.processor = preprocessor

def BLIP_BASELINE():
    from BLIP.models.blip_itm import blip_itm

    model = blip_itm(pretrained='./BLIP/chatir_weights.ckpt',  
                     med_config='./BLIP/configs/med_config.json',
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

class BLIPImageRetrieval:
    def __init__(self, clip_model_id=None, faiss_index_path='image_index.faiss', image_filenames_path='image_filenames.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载BLIP模型和图像嵌入器
        self.text_embedder, self.image_embedder = BLIP_BASELINE()

        # 加载FAISS索引
        self.index = faiss.read_index(faiss_index_path)
        # 加载图像文件名
        with open(image_filenames_path, 'rb') as f:
            self.image_filenames = pickle.load(f)

    def retrieve_images(self, text, top_k=5):
        """ Retrieve top K similar images for a given text query """
        # Tokenize and prepare the input for the model
        text_input = self.text_embedder.tokenizer(text, padding='longest', truncation=True, max_length=300, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 对文本查询进行编码
            text_output = self.text_embedder.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True, mode='text')
            text_embedding = self.text_embedder.text_proj(text_output.last_hidden_state[:, 0, :]).cpu().detach().numpy()
        
        # 使用FAISS进行相似性搜索
        distances, indices = self.index.search(text_embedding, top_k)
        similar_image_paths = [self.image_filenames[idx] for idx in indices[0]]
        return similar_image_paths, distances[0], indices[0]

class RelevanceFeedback:
    """BLIP"""
    def __init__(self, ImageRetrieval,
                 image_embedding_path=None):
        self.vlm = ImageRetrieval
        with open(image_embedding_path, 'rb') as f:
            self.image_embeddings = pickle.load(f)
        
    def provide_feedback(self, retrieved_images, retrieved_image_index, target_image_path):
        """
        给定一个查询和一个图像文件名列表，使用FAISS计算相似性,
        检索到的图像与目标图像之间，并返回相似度最高的图像。
        """
        image = self.vlm.image_embedder.processor(target_image_path).unsqueeze(0).to(self.vlm.text_embedder.text_encoder.device)
        with torch.no_grad():
            target_image_embedding = self.vlm.image_embedder.model(image).cpu().detach().numpy()
            target_image_embedding /= np.linalg.norm(target_image_embedding, ord=2, axis=1, keepdims=True) 
        
        retrieved_image_embedding = self.image_embeddings[retrieved_image_index]
        similarities = np.dot(retrieved_image_embedding, target_image_embedding.T).flatten() 
        
        # 找到相似度最高的索引
        best_match_index = list(np.argsort(similarities)[-5:][::-1])
        best_match_images = [self.vlm.image_filenames[retrieved_image_index[i]] for i in best_match_index]        
        return list(best_match_images), list(similarities[best_match_index])

class Captioning:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                quantization=True):
        # TODO:初始化LLaVA
        quantization_config = BitsAndBytesConfig(
              load_in_8bit=True,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, 
                                                                       torch_dtype=torch.float16, 
                                                                       quantization_config=quantization_config if quantization else None,
                                                                       attn_implementation="flash_attention_2",
                                                                       low_cpu_mem_usage=True,
                                                                       device_map="auto")  # 加载图像描述模型
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

    # 为选中的图像生成简短描述
    def generate_caption(self, image_path):
        # prompt = f"""Describe the image in more details. You can follow the 3 steps. You will extract the triplets from image 
        # but not output the triplets. you only output the description of the image. 
        # **Instruction**
        # 1. Extract all the entities from the images.
        # 2. Extract the relationship among entities, and then form the triplet (e.g. (entity_1, relationship, entity_2))
        # 3. Describe the images based on the triplet you extracted and your knowledge.
        # Extracted triplets:
        # Output your description: 
        # """
        # prompt = f"""Describe the image in more details. """
        # prompt = f"""Describe the image as detailed as possible""" # the prompt for clip with the best performance
        prompt = f"""Describe the image within 1 sentence"""

        conversation = [{"role": "user",
                          "content": [
                              {"type": "text", "text": f"{prompt}"},
                              {"type": "image"},
                            ]}]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image_path)
        tokens = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        generate_ids = self.model.generate(**tokens, 
                                           max_new_tokens=200,
                                           eos_token_id = self.processor.tokenizer.eos_token_id,
                                           pad_token_id = self.processor.tokenizer.pad_token_id,
                                            temperature=0,
                                           do_sample=False,
                                          )
        caption = self.processor.batch_decode(generate_ids, skip_special_tokens=True, 
                                              # clean_up_tokenization_spaces=False
                                             )[0]
        caption = caption.split('[/INST]')[-1]
        return caption

class QueryEditing:
    def __init__(self, model_id="/data/wangbin/No3/LLaMA_SFT/LLaMA_3.1_8B/Meta-Llama-3.1-8B-Instruct/",
                quantization=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)  # 加载文本处理器
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        quantization_config = BitsAndBytesConfig(
              load_in_8bit=True,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          torch_dtype=torch.float16, 
                                                          quantization_config=quantization_config if quantization else None,
                                                          attn_implementation="flash_attention_2",
                                                          device_map="auto")  # 加载文本生成模型

    def edit_query(self, query, caption):
        prompt = f"""Rewrite the query by incorporating the caption into the query either as an adjective clause, 
relative clause, or with-structure to make it vivid, simple, and descriptive.
                    query: '{query}'
                    caption: '{caption}'
                    
<Your revised only one query with the caption included as a descriptive clause or with-structure within 60 words>
                    """
        messages = [{"role": "user", "content": f"{prompt}"}]
        tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
        tokens = self.tokenizer(tokens, return_tensors='pt').to(self.model.device)
        generate_ids = self.model.generate(**tokens, max_new_tokens=100, do_sample=True,
                                          eos_token_id = self.tokenizer.eos_token_id,
                                           pad_token_id = self.tokenizer.pad_token_id,
                                            temperature=0.1,
                                          )
        new_query = self.tokenizer.batch_decode(generate_ids, 
                                                # skip_special_tokens=True, 
                                                # clean_up_tokenization_spaces=False
                                               )[0]
        new_query = new_query.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        new_query = new_query.replace('<|eot_id|>', '')
        return new_query

def find_target_image_index(arr, target):
    try:
        return arr.index(target)
    except ValueError:
        return -1

def compute_recall_at_top10(data):
    init_hit = 0
    round1_hit = 0 
    for record in data:
        image_ranking = record['image_ranking']
        if image_ranking < 10:
            init_hit += 1
        new_image_ranking = record['new_image_ranking']
        if new_image_ranking < 10:
            round1_hit += 1
    init_recall_at_topk = init_hit / len(data)
    round1_recall_at_topk = round1_hit /  len(data)
    return init_recall_at_topk, round1_recall_at_topk

base_image_dir = './flickr30k/flickr30k_images/'

image_retrieval_model = BLIPImageRetrieval(clip_model_id=None, 
                 faiss_index_path='./checkpoints/m_query_expansion_faiss.index', 
                 image_filenames_path='./checkpoints/m_query_expansion_id2image_path.pkl')

query_edit_model = QueryEditing(model_id="/data/wangbin/No3/LLaMA_SFT/LLaMA_3.1_8B/Meta-Llama-3.1-8B-Instruct/", quantization=True)

feedback_model = RelevanceFeedback(image_retrieval_model,
                                   # image_embedding_path='./checkpoints/clip_image_embedding_flickr30k.pickle'
                                   # image_embedding_path='./checkpoints/blip_image_embedding_flickr30k.pickle'
                                   image_embedding_path = './checkpoints/m_query_expansion_image_embedding.pkl'
                                  )

caption_model = Captioning(model_id="llava-hf/llava-v1.6-mistral-7b-hf", quantization=True)

def multi_modal_query_expansion(record, output_file_path):
    refine_query = None
    query = record['comment']
    # print('-'*100)
    # print('query', query)
    target_image_path = os.path.join(base_image_dir, record['image_name'])
    # print('target_image_path', target_image_path)
    # show_image([target_image_path], sentence='target image')
    refine_queries = [query]
    image_rankings = []
    new_image_rankings = []
    relevant_images = []
    relevant_cos_sims = []
    captions = []
    retrieved_imagess = []
    retrieved_cos_sims = []
    visited_relvant_images = set()
    for round in range(4):
        # 使用原始查询的初始检索
        topk = 50
        retrieved_images_all, retrieved_cos_sim_all, retrieved_indices_all = image_retrieval_model.retrieve_images(query, top_k=30000)
        retrieved_images, retrieved_cos_sim, retrieved_indices = retrieved_images_all[:topk], retrieved_cos_sim_all[:topk], retrieved_indices_all[:topk]
        retrieved_imagess.append(retrieved_images)
        retrieved_cos_sims.append(retrieved_cos_sim)
        # print('retrieved_images', retrieved_images)
        # print('retrieved_cos_sim', retrieved_cos_sim)
        # show_image(retrieved_images, sentence='Initial Retrieved images')

        # 检查目标图像是否落入检索到的前10张图像
        image_ranking = find_target_image_index(retrieved_images_all, target_image_path)
        image_rankings.append(image_ranking)
        # print('image_rankings', image_rankings)
        if 0 <= image_ranking < 10:
            break
        elif len(image_rankings) > 1:
            if (image_rankings[-1] - image_rankings[-2] > 0 ):
                break

        # 查询扩展
        # 相关性反馈
        relevant_images, relevant_cos_sims = feedback_model.provide_feedback(retrieved_images, retrieved_indices, target_image_path)
        # 从5张检索到的图像中选择1张图像
        while relevant_images:
            relevant_image = relevant_images.pop(0) # 弹出第一个图片
            relevant_cos_sim = relevant_cos_sims.pop(0)
            if relevant_image not in visited_relvant_images: # 检查这个图片是否已经访问过
                # 如果没有访问过，记录图片及其相似度
                relevant_images.append(relevant_image)
                relevant_cos_sims.append(relevant_cos_sim)
                # 将该图片添加到已访问集合中
                visited_relvant_images.add(relevant_image)
                # 找到未访问过的图片，跳出循环
                break
        # show_image([relevant_image], sentence ='Relevant image selected')

        # 为相关图像生成标题
        caption = caption_model.generate_caption(relevant_image)
        captions.append(caption)
        # print('-'*100)
        # print(f"Generated caption: {caption}")

        # 使用LLM优化查询
        refine_query = query_edit_model.edit_query(query, caption)
        refine_queries.append(refine_query)
        # print('-'*100)
        # print(f"New Query: {refine_query}")
        query = refine_query # update the query

    # 在循环结束后，写入结果到 JSON Lines 文件
    # output_file_path = './mm_rewrite_res/finetuned_blip_result.jsonl'  # 你可以根据需要修改文件路径
    # os.makedirs(output_file_path, exist_ok=True)  # 检查并创建目录

    
    with open(output_file_path, 'a+') as f:
        # 创建一个字典来保存结果
        write_record = {
        'original_query': record['comment'],
        'final_query': query,
        'refined_query': refine_query,
        'target_image_path': target_image_path,
        'image_rankings': image_rankings,
        'retrieved_images': retrieved_imagess,  # 转换 NumPy 数组为列表
        'retrieved_cos_sims': [cos_sim.tolist() for cos_sim in retrieved_cos_sims],
        'captions': captions,
        'relevant_images': relevant_images
    }
        
        # 将字典写入文件，使用 JSON 格式
        f.write(json.dumps(write_record) + '\n')  # 添加换行符以便于 JSON Lines 格式

dataset = pd.read_csv('./flickr30k/images_captions.csv')
# dataset = dataset.sample(frac=0.1, random_state=42) # downsample
data = dataset.to_dict(orient='records')
print(len(data))
query_target_pairs = {item['comment']: item['image_name'] for item in data}
# query_target_pairs

start_index = 0
for idx in tqdm(range(start_index, len(data))):
    record = data[idx]
    multi_modal_query_expansion(record, output_file_path = './mm_rewrite_res/finetuned_blip_result.jsonl' )
