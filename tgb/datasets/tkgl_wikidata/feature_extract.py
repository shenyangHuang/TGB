from transformers import RobertaModel, RobertaTokenizer
from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.datavalue import get_datavalue_from_snak_dict, WikibaseEntityId
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset, DataLoader
import argparse

class DescriptionDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename, dtype={'id': str, 'desc': str})
        self.title = df['title']
        self.desc = df['desc']
        self.id = df['id']

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        if isinstance(self.desc[idx], str):
            return self.id[idx], f'{self.title[idx]}, {self.desc[idx]}'
        else:
            return self.id[idx], ''

def parse_description():
    # Given wiidata dump, we extract its text descriptions
    wjd_dump_path = os.path.join("dump", "wikidata-20210517-all.json.gz")
    wjd = WikidataJsonDump(wjd_dump_path)

    entity_list = []
    entity_desc_list = []
    entity_title_list = []

    relation_list = []
    relation_desc_list = []
    relation_title_list = []

    for i, entity_dict in enumerate(tqdm(wjd, total=90121908)):
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
        elif entity_dict["type"] == "property":
            entity = WikidataProperty(entity_dict)
        else:
            continue

        entity_id = entity.entity_id
        desc = entity.get_description()
        title = entity.get_label()
        # print(title)
        
        if entity_dict["type"] == "item":
            assert(entity_id[0] == 'Q')
            entity_list.append(entity_id)
            entity_desc_list.append(desc)
            entity_title_list.append(title)
        elif entity_dict["type"] == "property":
            assert(entity_id[0] == 'P')
            relation_list.append(entity_id)
            relation_desc_list.append(desc)
            relation_title_list.append(title)

    dir_name = 'description'
    os.makedirs(dir_name, exist_ok = True)

    pd.DataFrame({'id': entity_list, 'title': entity_title_list, 'desc': entity_desc_list}).to_csv(os.path.join(dir_name, 'entity.csv'), index = False)
    pd.DataFrame({'id': relation_list, 'title': relation_title_list, 'desc': relation_desc_list}).to_csv(os.path.join(dir_name, 'relation.csv'), index = False)

def get_mpnet_embedding():
    parser = argparse.ArgumentParser(description='get mpnet embeddings from wikidata descriptions')
    parser.add_argument('--device', type=int, default=3)
    args = parser.parse_args()
    print(args)

    dir_name = 'description'

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    embedder = SentenceTransformer('paraphrase-mpnet-base-v2').to(device)

    relation_dataset = DescriptionDataset(os.path.join(dir_name, 'relation.csv'))
    relation_loader = DataLoader(relation_dataset, batch_size=512, shuffle=False, num_workers = 8)
    relation_emb_list = []

    for data in tqdm(relation_loader):
        id, desc = data
        with torch.no_grad():
            emb = embedder.encode(list(desc), device = device)
        relation_emb_list.append(emb)
    
    relation_emb_mat = np.concatenate(relation_emb_list, axis = 0).astype(np.float16)

    print(relation_emb_mat.shape)
    np.savez_compressed(os.path.join(dir_name, 'relation_mpnet_emb.npz'), emb = relation_emb_mat)
    
    entity_dataset = DescriptionDataset(os.path.join(dir_name, 'entity.csv'))
    entity_loader = DataLoader(entity_dataset, batch_size=512, shuffle=False, num_workers = 8)
    entity_emb_list = []

    for data in tqdm(entity_loader):
        id, desc = data
        with torch.no_grad():
            emb = embedder.encode(list(desc), device = device)
        entity_emb_list.append(emb)
    
    entity_emb_mat = np.concatenate(entity_emb_list, axis = 0).astype(np.float16)
    print(entity_emb_mat.shape)
    np.savez_compressed(os.path.join(dir_name, 'entity_mpnet_emb.npz'), emb = entity_emb_mat)


if __name__ == '__main__':
    # parse_description()
    get_mpnet_embedding()