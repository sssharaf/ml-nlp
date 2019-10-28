import json
import torch
from torch.utils.data import DataLoader,Dataset
import pytorch_transformers as pt
from pytorch_transformers import BertTokenizer, BertConfig,BertForMaskedLM,BertModel
import os
import typing
from typing import Dict,List,Sequence,Set
from types import SimpleNamespace as SN
T_BertTokenizer = typing.NewType("BertTokenizer",BertTokenizer)

EMPTY_COL='[EMPTY]'
def data_from_tables(tab_file:str,dir:str='wikisql/data') -> Dict[str,SN]:
    tab_map:Dict[str,SN]={}
    for l in open(dir + f'{os.sep}' + tab_file):
        l = json.loads(l.strip())
        header = l['header']
        header.append(EMPTY_COL)
        types= l['types']
        id=l['id']
        sn = SN()
        sn.header=header
        sn.types = types
        tab_map[id]=sn
    return tab_map

def data_from_sql(sql_file:str,dir='wikisql/data') -> (List[str], List[str],List[List[SN]]):
    sql_text: List[str]=[]
    table_id: List[str]=[]
    all_conds: List[List[SN]]=[]
    for l in open(dir + f'{os.sep}' + sql_file):
        l = json.loads(l.strip())
        tab_id = l['table_id']
        table_id.append(tab_id)
        sql_text.append(l['question'])
        sql = l['sql']
        conds=[]
        for cond in sql['conds']:
            sn = SN()
            sn.ci = cond[0]
            sn.oi = cond[1]
            sn.c = cond[2]
            conds.append(sn)
        all_conds.append(conds)
    return table_id,sql_text,all_conds


class MyDataSet(Dataset):

    def encode_headers(self,tabL, tokenizer: BertTokenizer) -> List[str]:
        enc_headers: List[str] = []
        for h in tabL['header']:
            enc_headers.extend(tokenizer.encode(h))
            enc_headers.extend(tokenizer.encode(tokenizer.sep_token))
        return enc_headers

    def __init__(self,sql_file='dev.jsonl',tab_file='dev.tables.jsonl',dir='wikisql/data'):
        super(MyDataSet).__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #tokenizer.add_special_tokens({'empty_col': EMPTY_COL})
        tab_map = data_from_tables(tab_file)
        table_ids, sql_texts, all_conds = data_from_sql(sql_file)
        data = []
        for tab_id,sql_text in zip(table_ids,sql_texts):
            tab_info = tab_map[tab_id]
            enc_s = tokenizer.encode(sql_text)
            enc_s.extend(tokenizer.encode(tokenizer.sep_token))
            enc_h=[]
            for h in tab_info.header:
                enc_h.extend(tokenizer.encode(h))
                enc_h.extend(tokenizer.encode(tokenizer.sep_token))
            enc_s.extend(enc_h)
            data.append(enc_s)

        max_l = 0
        for d in data:
            if len(d) > max_l:
                max_l= len(d)
        self.data = data
        self.max_l = max_l

    def __getitem__(self, idx):
        d = self.data[idx]
        X = torch.zeros(self.max_l,dtype=torch.long)
        X[:len(d)] = torch.tensor(d,dtype=torch.long)
        return X

    def __len__(self):
        return len(self.data)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
