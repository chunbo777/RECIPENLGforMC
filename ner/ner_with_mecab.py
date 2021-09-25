import os
import sys
import traceback
import json
import mysql.connector
from tqdm import tqdm
from konlpy.tag import Mecab
from datetime import datetime


class RecipeWithMySqlPipeline:
    def __init__(self):
        self.create_connection()
        self.create_table()
    
    def create_connection(self):
        
        nameserver = None
        if os.sys.platform =='win32':
            nameserver = 'localhost'
        elif os.sys.platform == 'linux':
            nameserver = '172.31.16.1'# /etc/resolv.conf
            

        mydb = mysql.connector.connect(
            charset='utf8'
            , db='Recipe'
            , host=nameserver
            , user="dasomoh"
            , password="1234"
        )
        self.conn = mydb
        self.curr = self.conn.cursor(buffered=True)
    
    def create_table(self):        
        sql ='''
        CREATE TABLE if not exists Recipe (
            Recipeid int NOT NULL AUTO_INCREMENT,
            title varchar(255) NOT NULL,
            link varchar(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (Recipeid)
        );
        '''
        self.curr.execute(sql)
        sql ='''
        CREATE TABLE if not exists Ingredients (
            IngrId int NOT NULL AUTO_INCREMENT,
            Recipeid int NOT NULL,
            ingredient varchar(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (IngrId),
            FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
        );
        '''
        self.curr.execute(sql)
        sql ='''
        CREATE TABLE if not exists Directions (
            Dirid int NOT NULL AUTO_INCREMENT,
            Recipeid int NOT NULL,
            direction varchar(1023),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (Dirid),
            FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
        );
        '''
        self.curr.execute(sql)
        sql ='''
        CREATE TABLE if not exists ner_mecab (
            id int NOT NULL AUTO_INCREMENT,
            Recipeid int NOT NULL,
            ner varchar(1023),
            pos varchar(128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
        );
        '''
        self.curr.execute(sql)

    def process_item(self, item):
        for k in ['title','ingredients','directions', 'ner_mecab']:
            if item[k] is None:
                break
                # if k == 'title':# 아예 내용이 없는 경우
                #     print(item)
                # else:
                #     continue

            if isinstance(item[k], str):
                item[k] = item[k].replace('"','').replace('\\','').strip()
            elif isinstance(item[k], list):
                item[k] = [(i[0].replace('"','').replace('\\','').strip(),i[1]) if isinstance(i, tuple) else i.replace('"','').replace('\\','').strip() for i in item[k]]

        try:
            self.store_db(item)
        except Exception:
            traceback.print_exc(file=open(os.path.dirname(__file__)+f'/log/{os.path.basename(__file__)}_log_{datetime.now().strftime("%Y%m%d")}.txt', mode='a'))
            print(item['title'])
            print(item['link'])

        return item

    def store_db(self, item):

        sql = f'''INSERT INTO Recipe (title, link) VALUES ("{item['title']}" ,"{item['link']}" );'''
        self.curr.execute(sql)

        # self.curr.execute('SELECT LAST_INSERT_ID()')
        recipeid = str(self.curr.lastrowid)

        for ingredient in item['ingredients']:
            sql = f' INSERT INTO Ingredients (Recipeid, ingredient) VALUES ("{recipeid}" ,"{ingredient}");'
            self.curr.execute(sql)
        
        for (i, direction) in enumerate(item['directions']):
            sql = f'INSERT INTO directions (Recipeid, direction) VALUES ("{recipeid}","{direction}");'
            try:
                self.curr.execute(sql)
            except Exception:
                if i>0: 
                    splitted_direction = direction.split(item['directions'][i-1])[-1]# 동일한 내용이 중복으로 들어가 내용이 너무 긴경우
                    try:
                        self.curr.execute(f'INSERT INTO directions (Recipeid, direction) VALUES ("{recipeid}","{splitted_direction}");')
                    except Exception:
                        traceback.print_exc(file=open(os.path.dirname(__file__)+f'/log/{os.path.basename(__file__)}_log_{datetime.now().strftime("%Y%m%d")}.txt', mode='a'))
        
        for ner_mecab in item['ner_mecab']:
            sql = f'INSERT INTO ner_mecab (Recipeid, ner, pos) VALUES ("{recipeid}","{ner_mecab[0]}","{ner_mecab[1]}");'
            self.curr.execute(sql)

        self.conn.commit()
    def automated_raw_tagging_item(self, target):
        sql = f'''select ner, pos from ner_set;'''
        self.curr.execute(sql)
        raw_ner_set = self.curr.fetchall()
        return raw_ner_set
            


# with open(f'/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/final.json', encoding='utf8') as f:
#     jsonString = '['+f.read().replace('}{','},{')+']'
#     jsondata = json.loads(jsonString)
#     # jsondata = json.load(fp=f)
#     # df = pd.read_json(f)

path = '/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/'
# with open(f'{path}recipes_in_korean.json', mode='w', encoding='utf8') as f:
#     json.dump(jsondata, fp=f)

me = Mecab()
sql = RecipeWithMySqlPipeline()
# with open(f'{path}recipes_in_korean.json', mode='r', encoding='utf8') as f:
#     jsondata = json.load(fp=f)

#     for data in tqdm(jsondata):
#         data['ner_mecab'] = list(set([tup for ingr in data['ingredients'] for tup in me.pos(ingr)]))
#         sql.process_item(data)
def process_text(repl_info, target):
    modified_target = []
    for i, repl in enumerate(repl_info['repl']):
        if i == 0:
            tmp =  target[:repl_info['loc'][i][0]]
            modified_target.extend([tmp, repl])
        elif i < len(repl_info['repl'])-1 and i >0:
            tmp = target[repl_info['loc'][i-1][1]: repl_info['loc'][i][0]]
            modified_target.extend([tmp, repl])
        elif i == len(repl_info['repl'])-1:
            tmp1 =  target[repl_info['loc'][i-1][1]: repl_info['loc'][i][0]]
            tmp2 =  target[repl_info['loc'][i][1]:]
            modified_target.extend([tmp1, repl,tmp2])
    modified_target = ' '.join(modified_target)
    return modified_target


import re
import pandas as pd
import numpy as np
from tqdm import tqdm
path = '/home/dasomoh88/RECIPENLGforMC/ner/data/'
with open(f'{path}0925_ner_298452_1.csv', mode='r', encoding='utf8') as f:

    df = pd.read_csv(f, encoding='utf8')
    data = df.to_numpy()

    tagged_data = []
    for row in tqdm(data):
        target = f' {row[0]} '
        unit = row[1].split('@#') if isinstance(row[1], str) else []
        qty = row[2].split('@#') if isinstance(row[2], str) else []
        
        regex_For_Qty = f'([\d]*[,|/|.]*[\d]*|({"|".join(qty)})*)'
        regex_For_Unit = f'(큰술|작은술|적량|약간{"|"+"|".join(unit)})'
        regex = f'{regex_For_Qty}[\s]*{regex_For_Unit}'

        repl_info= {'loc':[], 'repl':[]}
        for found in re.finditer(regex, target):
            if found is None or found.group().strip()=='':
                continue
            else:
                matched = found.group().strip()
            repl = ''
            for k, regex_ in {'QTY':regex_For_Qty, 'UNIT':regex_For_Unit}.items():
                matched_ = re.search(regex_, matched)
                if matched_ is not None and matched_.group().strip() !='':
                    matched__ = matched_.group().strip()
                    if matched__ !='':
                        repl += f'<{matched__}:{k}>'# 중복 대체 연산 방지
            repl_info['loc'].append(found.span())
            repl_info['repl'].append(repl)# 수량, 단위 정보 한번에

        modified_target = target if len(repl_info['repl']) ==0 else ' '+process_text(repl_info, target)

        ingr = row[3].split('@#') if isinstance(row[3], str) else []
        regex_For_Ingr = f'([^((<|>)(\w*|ㄱ-힣*))]*({"|".join(ingr)})(<\w*|ㄱ-힣*)*[^(:)])'
        repl_info= {'loc':[], 'repl':[]}
        for found in re.finditer(regex_For_Ingr, modified_target):
            if found is None or found.group().strip()=='':
                continue
            else:
                matched = re.sub('[^\w]', '', found.group())
                if matched !='':
                    repl = f'<{matched}:INGR>'
                    repl_info['loc'].append(found.span())
                    repl_info['repl'].append(repl)

        tagged_target = modified_target if len(repl_info['repl']) ==0 else process_text(repl_info, modified_target)
        if tagged_target == '':
            print(modified_target)
            print(tagged_target)
        else:        
            tagged_data.append([row[-1],tagged_target.replace('@#','')])

pd.DataFrame(tagged_data).to_csv(open(f'{path}0926_tagged.csv', mode='w', encoding='utf8'), header=False, index=False )

