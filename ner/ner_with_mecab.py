import os
import sys
import traceback
import json
import mysql.connector
from tqdm import tqdm
from konlpy.tag import Mecab
from datetime import datetime
import numpy as np

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

# DB에 데이터 저장
# me = Mecab()
# sql = RecipeWithMySqlPipeline()
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
from datetime import date
from tqdm import tqdm
from konlpy.tag import Mecab

def get_tagged_data(path, file_name):
    # with open(f'{path}0925_ner_298452_1.csv', mode='r', encoding='utf8') as f:
    with open(f'{path}{file_name}', mode='r', encoding='utf8') as f:


        df = pd.read_csv(f, encoding='utf8')
        data = df.to_numpy()

        tagged_data = []
        regex_val = []

        for row in tqdm(data):
            target = f' {row[0]} '
            # unit = [u if u not in ['장'] else f'{u}' for u in row[1].split('@#') ] if isinstance(row[1], str) else []
            unit = [u if u not in ['장'] else f'{u}' for u in row[-4].split('@#') ] if isinstance(row[-4], str) else []
            # qty = row[2].split('@#') if isinstance(row[2], str) else []
            qty = row[-3].split('@#') if isinstance(row[-3], str) else []
            
            regex_For_Qty = f'([\d\s]+[,|/|.]*[\d]*|({"|".join(qty)})+)'#수량
            regex_For_Unit = f'(큰술|작은술{"|"+"|".join(unit)})'#단위
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
                repl_info['loc'].append(found.span())#국간장 = 국간+장
                repl_info['repl'].append(repl)# 수량, 단위 정보 한번에

            if len(repl_info['repl']) ==0:
                modified_target = target
                regex_val.append([target, regex, row[1]])# 정규식 검정용
            else:
                modified_target = ' '+process_text(repl_info, target)

            # ingr = row[3].split('@#') if isinstance(row[3], str) else []
            ingr = row[-2].split('@#') if isinstance(row[-2], str) else []# 식재료
            ingr = [i if i not in ['파', '마늘', '생강'] else f'(다진\s|)*{i}' for i in ingr]
            ingr = [i if i not in ['고추'] else f'(붉은\s|풋\s)*{i}' for i in ingr]
            ingr = [i if i not in ['미역', '쇠미역', '문어'] else f'(마른\s|)*{i}' for i in ingr]
            # if '찹쌀' in ingr:
            #     print(ingr)
            ingr = [i if i not in ['전분가', '고추가', '콩가'] else f'{i}(루\s|)*' for i in ingr]
            ingr = [i if i not in ['찹쌀'] else f'{i}(가루\s|)*' for i in ingr]
            ingr = [i if i not in ['토끼'] else f'{i}(고기\s|)*' for i in ingr]
            ingr = [i for i in ingr if i not in ['적량']]# 감성돔 추가필요
            # regex_For_Ingr = f'([^((<|>)(\w*|ㄱ-힣*))]*({"|".join(ingr)})(<\w*|ㄱ-힣*)*[^:])'

            regex_For_Ingr = f'((\w*|ㄱ-힣*)*({"|".join(ingr)})(\w*|ㄱ-힣*)*)'

            repl_info= {'loc':[], 'repl':[]}
            for found in re.finditer(regex_For_Ingr, modified_target):
                if found is None or found.group().strip()=='':
                    continue
                elif re.search('[<][\w|ㄱ-힣]+[:]', found.group().strip()) is not None:
                    continue# 이미 TAGGING 된 정보는 제외
                else:
                    matched = re.sub('[^\w]', '', found.group())
                    if matched !='':
                        repl = f'<{matched}:INGR>'
                        repl_info['loc'].append(found.span())
                        repl_info['repl'].append(repl)
            if len(repl_info['repl']) ==0:
                tagged_target = modified_target
                regex_val.append([modified_target, regex, row[1]])# 정규식 검정용
            else:
                tagged_target = process_text(repl_info, modified_target)

            if tagged_target == '':
                continue
            else:        
                tagged_data.append([row[-1],row[0],tagged_target.replace('@#','')])

    m = Mecab()
    ## insert 구문 생성
    ## 식재료가 제대로 tagging 되지 않은 경우

    ingr_to_add = [ ]
    for i, k in np.array(regex_val)[:,[0,-1]]:
        if ":UNIT" not in i or ":QTY" not in i:# test
            print(i)
        text = re.sub('(<[ㄱ-힣|\w|\d]*:[UNIT|QTY]*>|[(]|[)]|[<]|[>]|[방법]|[\s]|[\d]|[〈]|[〉]|[:])+',' ',i.strip()).replace(']','').replace('[','')
        for j in re.split(' ',text):
            if j not in ['',]:
                ingr_to_add.append(f"INSERT INTO ner_set (ner, pos, cate) values ('{j}','custom','ingr');##{i} #### {k}")

    ingr_set = set(ingr_to_add)# 중복제거
    pd.DataFrame(ingr_set).to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_ingr_to_add_{len(ingr_set)}.csv',mode='w', encoding='utf8'), header=False, index=False, sep='\t' )

    pd.DataFrame(tagged_data).to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_tagged.csv', mode='w', encoding='utf8'), header=False, index=False )
    print(f'SAVED!! ######## {path}{datetime.today().strftime("%y%m%d%H")}_tagged.csv')
# path = f'{os.path.dirname(__file__)}/data/'
# get_tagged_data(path, '0925_ner_298452_1.csv')url


def get_BIO_data(path, data):
    modified_data = []
    for row in tqdm(data):
        target = row[-1]
        entities = ['INGR','QTY','UNIT']
        regex = f'[<]([\wㄱ-힣]|[\d\s]+[,|/|.]*[\d]*)+:({"|".join(entities)})[>]'
        repl_info = {'loc':[], 'repl':[]}

        for found in re.finditer(regex, target):
            if found is None or found.group().strip('<>')=='':
                continue
            else:
                matched = found.group().strip(' <>').split(':')# 0: 요소, 1: entity
                chars = list(matched[0])
                labels = []
                for i in range(len(chars)):
                    if i ==0:
                        labels.append(f'B-{matched[1]}')
                    else:
                        labels.append(f'I-{matched[1]}')
                repl_info['repl'].append([chars, labels])    
                repl_info['loc'].append(found.span())


        # 사례별
        modified_chars = []
        modified_labels = []
        for i, repl in enumerate(repl_info['repl']):
            if i == 0:
                tmp =  list(target[:repl_info['loc'][i][0]])

                tmp_lbl = ['O']*len(tmp)
                tmp_lbl.extend(repl[1])
                modified_labels.extend(tmp_lbl)

                tmp.extend(repl[0])
                modified_chars.extend(tmp)
            # elif i < len(repl_info['repl'])-1 and i >0:
            # elif 0<i and  i < len(repl_info['repl'])-1:
            elif 0 < i and i < len(repl_info['repl'])-1:
                tmp = list(target[repl_info['loc'][i-1][1]: repl_info['loc'][i][0]])

                tmp_lbl = ['O']*len(tmp)
                tmp_lbl.extend(repl[1])
                modified_labels.extend(tmp_lbl)

                tmp.extend(repl[0])
                modified_chars.extend(tmp)
                # modified_target.extend([tmp, repl])
            elif i == len(repl_info['repl'])-1:
                tmp1 =  list(target[repl_info['loc'][i-1][1]: repl_info['loc'][i][0]])
                tmp2 =  list(target[repl_info['loc'][i][1]:])

                tmp_lbl1 = ['O']*len(tmp1)
                tmp_lbl2 = ['O']*len(tmp2)
                repl[1].extend(tmp_lbl2)
                tmp_lbl1.extend(repl[1])
                modified_labels.extend(tmp_lbl1)

                repl[0].extend(tmp2)
                tmp1.extend(repl[0])
                modified_chars.extend(tmp1)
                # modified_target.extend([tmp1, repl,tmp2])
        # modified_target = ' '.join(modified_target)
        assert len(modified_chars) == len(modified_labels)
        modified_data.append([row[0], row[1], row[-1], [modified_chars, modified_labels]])

    # # 임시
    train = modified_data[:int(len(modified_data)*0.5)]
    val = modified_data[int(len(modified_data)*0.5):int(len(modified_data)*0.8)]
    test = modified_data[int(len(modified_data)*0.8):]
    data = {'train':train, 'val':val, 'test':test}
    # with open(f'{path}{datetime.today().strftime("%y%m%d%H")}_bio.json', 'w', encoding='utf8') as f:
    #     result = pd.DataFrame(modified_data).to_json(orient="values")
    #     parsed = json.loads(result)
    #     jsonData = json.dumps(parsed, indent=4, ensure_ascii=False)
    #     f.write(jsonData)
    # print(f'SAVED!! ######## {path}{datetime.today().strftime("%y%m%d%H")}_bio.json')

    for k, v in data.items():
        with open(f'{path}{datetime.today().strftime("%y%m%d%H")}_bio_{k}.tsv', mode='a', encoding='utf8') as f:
            for row in tqdm(v):
                for el in row:
                    f.write(f'\n')
                    if isinstance(el, list):
                        text = '\n'.join(['\t'.join(i) for i in zip(el[0], el[1])])
                        f.write(text)
                    elif isinstance(el, str):
                        el = el.replace('\n','')
                        f.write(f'##{el}')
                    elif isinstance(el, int):
                        f.write(f'##{str(el)}')
        print(f'SAVED!! ######## {path}{datetime.today().strftime("%y%m%d%H")}_bio_{k}.tsv')

path = f'{os.path.dirname(__file__)}/data/'
get_tagged_data(path, 'test_data_ner_1000.csv')
# df_to_bio = pd.read_csv(f'{path}21092820_tagged.csv', encoding='utf8')
# data = df_to_bio.to_numpy() 
# get_BIO_data(path, data)#f'{path}{datetime.today().strftime("%y%m%d%H%M")}_bio.tsv'

