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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

max_hs_len = 3

def data_from_tables(tab_file:str,dir:str='wikisql/data') -> Dict[str,SN]:
    tab_map:Dict[str,SN]={}
    for l in open(dir + f'{os.sep}' + tab_file):
        l = json.loads(l.strip())
        header = l['header']
        header.append(EMPTY_COL)
        e_header = [tokenizer.encode(h) for h in header]
        types = l['types']
        id = l['id']
        sn = SN()
        sn.header = header
        sn.e_header = e_header
        sn.types = types
        tab_map[id]=sn
    return tab_map

def data_from_sql(sql_file:str,dir='wikisql/data') -> (List[str], List[str],List[str],List[List[SN]]):
    sql_text: List[str]=[]
    table_id: List[str]=[]
    all_conds: List[List[SN]]=[]
    for l in open(dir + f'{os.sep}' + sql_file):
        l = json.loads(l.strip())
        tab_id = l['table_id']
        table_id.append(tab_id)
        sql_text.append(l['question'])
        sql = l['sql']
        conds = []
        for cond in sql['conds']:
            sn = SN()
            sn.ci = cond[0]
            sn.oi = cond[1]
            sn.c = cond[2]
            conds.append(sn)
        all_conds.append(conds)
    e_sql_text = [tokenizer.encode(s) for s in sql_text]
    return table_id,sql_text,e_sql_text, all_conds


tab_map = data_from_tables('dev.tables.jsonl')
#max_hs_len = max([len(ehs) for sn in tab_map.values() for ehs in sn.e_header])


class MyDataSet(Dataset) :
    def __init__(self,max_hs_len=3,sql_file='dev.jsonl',tab_file='dev.tables.jsonl',dir='wikisql/data'):
        super(MyDataSet).__init__()
        tab_map = data_from_tables(tab_file)
        table_ids, sql_texts,e_sql_text, all_conds = data_from_sql(sql_file)
        data = []

        for tab_id,enc_s in zip(table_ids,e_sql_text):
            x = []
            x.extend(enc_s)
            tab_info = tab_map[tab_id]
            x.extend(tokenizer.encode(tokenizer.sep_token))
            for enc_h in tab_info.e_header:
                enc_h = enc_h[:max_hs_len]
                x.extend(enc_h)
                x.extend(tokenizer.encode(tokenizer.sep_token))
            data.append(x)

        max_l = 0
        for d in data:
            if len(d) > max_l:
                max_l = len(d)
        self.data = data
        self.max_l = max_l
        self.all_conds = all_conds

    def __getitem__(self, idx):
        d = self.data[idx]
        X = torch.zeros(self.max_l,dtype=torch.long)
        X[:len(d)] = torch.tensor(d,dtype=torch.long)

        Y = torch.tensor(len(self.all_conds[idx]),dtype=torch.long)
        return X,Y

    def __len__(self):
        return len(self.data)
