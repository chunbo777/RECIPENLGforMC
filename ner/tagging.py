import os
import sys
import traceback
import json
import mysql.connector
from tqdm import tqdm
from konlpy.tag import Mecab
from datetime import datetime
import numpy as np
import json
import re

class RecipeWithMySqlPipeline:
    def __init__(self):
        self.create_connection()
        # self.create_table()
    
    def create_connection(self):
        
        nameserver = 'localhost'
        if os.sys.platform =='win32':
            nameserver = 'localhost'
        elif os.sys.platform == 'linux':
            nameserver = '172.30.176.1'# /etc/resolv.conf
            

        mydb = mysql.connector.connect(
            charset='utf8'
            , db='recipe'
            # , host=nameserver
            , host='3.37.218.4', port = '3306'# team서버에서는 연결이 안됨#Authentication plugin 'caching_sha2_password' cannot be loaded
            # , user="dasomoh"
            , user="aws_mysql_lab17"# 연결이 안됨
            , password="1234"
        )
        self.conn = mydb
        self.curr = self.conn.cursor(buffered=True)
    
    def create_table(self):        
        # sql ='''
        # CREATE TABLE if not exists Recipe (
        #     Recipeid int NOT NULL AUTO_INCREMENT,
        #     title varchar(255) NOT NULL,
        #     link varchar(255),
        #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #     PRIMARY KEY (Recipeid)
        # );
        # '''
        # self.curr.execute(sql)
        # sql ='''
        # CREATE TABLE if not exists Ingredients (
        #     IngrId int NOT NULL AUTO_INCREMENT,
        #     Recipeid int NOT NULL,
        #     ingredient varchar(255),
        #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #     PRIMARY KEY (IngrId),
        #     FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
        # );
        # '''
        # self.curr.execute(sql)
        # sql ='''
        # CREATE TABLE if not exists Directions (
        #     Dirid int NOT NULL AUTO_INCREMENT,
        #     Recipeid int NOT NULL,
        #     direction varchar(1023),
        #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #     PRIMARY KEY (Dirid),
        #     FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
        # );
        # '''
        # self.curr.execute(sql)
        sql ='''
        CREATE TABLE if not exists temp_ner (
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
       
        for k in ['title', 'link','ingredients','directions','NER']:
            if item[k] is None:
                break
                # if k == 'title':# 아예 내용이 없는 경우
                #     print(item)
                # else:
                #     continue

            # if isinstance(item[k], str):
            #     item[k] = item[k].replace('"','').replace('\\','').strip()
            # elif isinstance(item[k], list):
            #     item[k] = [(i[0].replace('"','').replace('\\','').strip(),i[1]) if isinstance(i, tuple) else i.replace('"','').replace('\\','').strip() for i in item[k]]

            if k in ['title', 'link']:
                item[k] = item[k].replace('"','').replace('\\','').strip()
            elif k in ['ingredients','directions','NER']:
                item[k] = json.loads(item[k])

        try:
            if item[item.axes[0][0]] >= 1411160:
                
                self.store_db(item)
        except Exception:
            print(item['title'])
            print(item['link'])
            print('#####3',item[item.axes[0][0]])
            print(traceback.print_exc())
            # traceback.print_exc(file=open(os.path.dirname(__file__)+f'/data/{os.path.basename(__file__)}_log_{datetime.now().strftime("%Y%m%d")}.txt', mode='a'))

        return item

    def store_db(self, item):
        sql = f'''INSERT INTO recipe (title, link) VALUES ("{item['title']}" ,"{item['link']}" );'''
        self.curr.execute(sql)

        # self.curr.execute('SELECT LAST_INSERT_ID()')
        recipeid = str(self.curr.lastrowid)

        for ingredient in item['ingredients']:
            ingredient = re.sub(r'([\\]|["])','',ingredient)
            sql = f' INSERT INTO ingredients (Recipeid, ingredient) VALUES ("{recipeid}" ,"{ingredient}");'
            self.curr.execute(sql)
        
        for (i, direction) in enumerate(item['directions']):
            direction = re.sub(r'([\\]|["])','',direction)
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
        
        for ner in item['NER']:
            ner = re.sub(r'([\\]|["])','',ner)
            sql = f'INSERT INTO temp_ner (Recipeid, ner) VALUES ("{recipeid}","{ner}");'
            self.curr.execute(sql)

        self.conn.commit()

    def store_set(self, target):
        for word in tqdm(target):
            word = re.sub(r'([\\]|["])','',word)
            sql = f'INSERT INTO words_for_tagging (word, cate, pos) VALUES ("{word}","ingr","NER_eng");'
            self.curr.execute(sql)
        self.conn.commit()

    
    def data_to_tag(self, limit =None):
        lim = ''
        if limit is not None :
            lim = f' limit {str(limit)}'
        
        sql = f'''
        with 
        A as (
            select max(Recipeid) Recipeid, title, link from recipe 
            where created_at < 20211006 and in_use =1 
            group by title, link  order by Recipeid desc {lim}
            )
        , B as ( select group_concat(B.ingredient  SEPARATOR ' ') ingr, A.Recipeid from A left join ingredients B on A.Recipeid = B.Recipeid where B.in_use =1 group by A.Recipeid)
        , C as ( select group_concat(B.direction  SEPARATOR '@#') dir, A.Recipeid from A left join directions B on A.Recipeid = B.Recipeid where B.in_use =1 group by A.Recipeid)
        , D as ( select A.Recipeid, B.word, B.cate, B.pos from rel_btw_recipe_and_ner A left join words_for_tagging B on A.NER_id = B.id where B.in_use =1 )
        , ingr as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where  D.cate = 'ingr' group by A.Recipeid)
        , unit as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where D.cate = 'unit' group by A.Recipeid)
        , qty as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where D.cate = 'qty' group by A.Recipeid)
        select A.title, A.link, B.ingr, C.dir, unit.word unit, qty.word qty, ingr.word ingr, B.Recipeid 
        from  A
        left join B on A.Recipeid = B.Recipeid
        left join C on A.Recipeid = C.Recipeid
        left join ingr on A.Recipeid = ingr.Recipeid
        left join unit on A.Recipeid = unit.Recipeid
        left join qty on A.Recipeid = qty.Recipeid
        where B.ingr is not null 
        -- and ingr.word is not null
        '''

        # sql = f'''
        # with 
        # A as (
        #     select max(Recipeid) Recipeid, title, link from recipe 
        #     where created_at < 20211006 and in_use =1
        #     group by title, link {lim}
        #     )
        # , B as ( select group_concat(B.ingredient  SEPARATOR '@#') ingr, A.Recipeid from A left join ingredients B on A.Recipeid = B.Recipeid where B.in_use =1 group by A.Recipeid)
        # , C as ( select group_concat(B.direction  SEPARATOR '@#') dir, A.Recipeid from A left join directions B on A.Recipeid = B.Recipeid where B.in_use =1 group by A.Recipeid)
        # , D as ( select A.Recipeid, B.word, B.cate, B.pos from rel_btw_recipe_and_ner A left join words_for_tagging B on A.NER_id = B.id where B.in_use =1 )
        # -- , ingr as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where  D.cate = 'ingr' group by A.Recipeid)
        # , unit as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where D.cate = 'unit' group by A.Recipeid)
        # , qty as ( select group_concat(D.word SEPARATOR '@#') word, A.Recipeid from A left join D on A.Recipeid = D.Recipeid where D.cate = 'qty' group by A.Recipeid)
        # select A.title, A.link
        # , B.ingr
        # , C.dir, unit.word unit, qty.word qty
        # , "" ingr
        # -- , ingr.word ingr
        # , B.Recipeid 
        # from  A
        # left join B on A.Recipeid = B.Recipeid
        # left join C on A.Recipeid = C.Recipeid
        # -- left join ingr on A.Recipeid = ingr.Recipeid
        # left join unit on A.Recipeid = unit.Recipeid
        # left join qty on A.Recipeid = qty.Recipeid
        # where B.ingr is not null 
        # -- and A.Recipeid > 160117
        # -- and ingr.word is not null 
        # ;
        # '''

        self.curr.execute(sql)
        data_to_tag = self.curr.fetchall()
        return data_to_tag

    def words_for_tagging(self):
        sql = f'''
        select id,REGEXP_REPLACE(word, '[\\s]+', ' ') word, cate, created_at 
        from words_for_tagging 
        where in_use =1
        -- and cate !='ingr' or (cate = 'ingr' and pos like 'fooddb%')
        ; 
        '''

        self.curr.execute(sql)# query 수행
        words_for_tagging = self.curr.fetchall()# query 결과 추출
        return words_for_tagging
    
    
            


# with open(f'/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/final.json', encoding='utf8') as f:
#     jsonString = '['+f.read().replace('}{','},{')+']'
#     jsondata = json.loads(jsonString)
#     # jsondata = json.load(fp=f)
#     # df = pd.read_json(f)

# path = '/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/'
# with open(f'{path}recipes_in_korean.json', mode='w', encoding='utf8') as f:
#     json.dump(jsondata, fp=f)

# DB에 데이터 저장
# me = Mecab()
# sql = RecipeWithMySqlPipeline()
import pandas as pd
# df = pd.read_csv('/home/tutor/lab17/RECIPENLGforMC/ner/data/full_dataset.csv')
# ners = set([ner for ners in df['NER'].apply(lambda x: json.loads(x)).values for ner in ners])
# sql.store_set(ners)
# df.iloc[1411166:].apply(lambda x: sql.process_item(x), axis=1)
# with open(f'{path}recipes_in_korean.json', mode='r', encoding='utf8') as f:
#     jsondata = json.load(fp=f)
#     for data in tqdm(jsondata):
#         # data['ner_mecab'] = list(set([tup for ingr in data['ingredients'] for tup in me.pos(ingr)]))
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
    modified_target = ''.join(modified_target)
    return modified_target


import re
import pandas as pd
from datetime import date
from tqdm import tqdm
from konlpy.tag import Mecab

def tag_data(data, words_for_tagging = None, target_index=2):
    tagged_data = []
    regex_val = []
    
    for row in tqdm(data):
        # target = f' {row[target_index]} '# 0: title, 1: url 2: ingredient, 3: directions
        # unit = [u if u not in ['장'] else f'{u}' for u in row[1].split('@#') ] if isinstance(row[1], str) else []
        unit = row[-4].split('@#') if isinstance(row[-4], str) else []
        # qty = row[2].split('@#') if isinstance(row[2], str) else []
        qty = row[-3].split('@#') if isinstance(row[-3], str) else []

        for target in row[target_index].split('@#'):# 0: title, 1: url 2: ingredient, 3: directions
            target = f' {target} '
            regex_For_Qty = f'(([\d]+[\s]?([,|/|.][\d])*|[\d]*[\s]?[\u2150-\u215E\u00BC-\u00BE])|({"|".join(qty)  if len(qty) >0 else "@#$" }))+'#수량
            # regex_For_Unit = f'(큰술|작은술{"|"+"|".join(unit)})'#단위
            regex_For_Unit = f'({"|".join(unit) if len(unit) >0 else "@#$" })'#단위
            regex = f'{regex_For_Qty}[\s]*{regex_For_Unit}[\s]?'
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
                    else:
                        print()
                repl_info['loc'].append(found.span())#국간장 = 국간+장
                repl_info['repl'].append(repl)# 수량, 단위 정보 한번에
            if len(repl_info['repl']) ==0:
                modified_target = target
                # regex_val.append([target, regex, row[1], row[-1],'QtyAndUnit'])# 정규식 검정용
            else:
                modified_target = ' '+process_text(repl_info, target)
            ingr = row[-2].split('@#') if isinstance(row[-2], str) else []# 식재료
            # regex_For_Ingr = f'((\w*|ㄱ-힣*)*({"|".join(ingr)})(\w*|ㄱ-힣*)*)'
            regex_For_Ingr = f'({"|".join(ingr) if len(ingr) > 1 or (len(ingr)==1 and ingr[0].strip() !="") else "@#"})+'
            repl_info= {'loc':[], 'repl':[]}
            if row[-1] == 6853:
                print()

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
            if len(repl_info['repl']) !=0:
                tagged_target = process_text(repl_info, modified_target)
            else:
                tagged_target = modified_target
            #     regex_val.append([modified_target, regex_For_Ingr, row[1], row[-1],'INGR'])# 정규식 검정용
            if tagged_target == '':
                continue
            else:# tagging이 된 data에서 추가로 tagging 할 여지가 있는지 확인
                regex_val.append([re.sub('[<][^<]+[:](INGR|UNIT|QTY)+[>]','@#',tagged_target), regex_For_Ingr, row[1], row[-1],'INGR'])# 정규식 검정용
                # regex_val.append([target, regex_For_Ingr, row[1], row[-1],'INGR'])# 정규식 검정용
                tagged_data.append([row[-1],target,tagged_target.replace('@#','')])
    m = Mecab()
    ## insert 구문 생성
    ## 식재료가 제대로 tagging 되지 않은 경우

    ingr_to_add_on_set = [ ]
    relation_info = [ ]
    relations = []
    for target,regex__, url, recipe_id, cate in tqdm(np.array(regex_val)):
        # continue
        # if int(recipe_id) == 6853:
        #     print()
        # else :
        #     continue
        # unit에서 새롭게 tagging 할 영역 탐색
        # units_for_tagging = words_for_tagging[words_for_tagging[:,2]=="unit"]
        # regex_For_Qty = f'([\d]+[\s]?([,|/|.][\d])*|[\d]*[\s]?[\u2150-\u215E\u00BC-\u00BE])+'#수량
        # regex_For_Unit =  f'({"|".join(sorted(set(units_for_tagging[:,1]), key=lambda unit: len(unit),reverse=True))})'#단위
        # regex = f'{regex_For_Qty}[\s]*{regex_For_Unit}[\s]?'
        # for found in re.finditer(regex, target.strip()):
        #     if found is None or found.group().strip()=='':
        #         continue
        #     else:
        #         matched = found.group().strip()
        #     for k, regex_ in {'UNIT':regex_For_Unit}.items():
        #         matched_ = re.search(regex_, matched)
        #         if matched_ is not None and matched_.group().strip() !='':
        #             matched__ = matched_.group().strip()
        #             if matched__ !='':
        #                 found_ = re.search(f'(\S)*.?{matched__}.?(\S)*', matched)
        #                 relation_info.append(
        #                     f"INSERT INTO rel_btw_recipe_and_ner (Recipeid, NER_id) values ({recipe_id}, {units_for_tagging[units_for_tagging[:,1]==matched__][:,0][0]});##{matched__}, {found_.group(0)}")

        # 식재료 부분에서 tagging 할 부분 탐색
        target = target.replace(']','').replace('[','').replace("'",'').strip()
        for info_for_tagging in sorted(words_for_tagging[words_for_tagging[:,2]=="ingr"], key=lambda x: len(x[1]), reverse=True):
            if info_for_tagging[1] in target.strip():
            # if info_for_tagging[1]==entity_candidate and info_for_tagging[2] not in ['ingr']:
                if re.search(f'(\s{info_for_tagging[1]}|{info_for_tagging[1]}\s)', target) is None: continue
                found = re.search(f'(\S)*.?{info_for_tagging[1]}.?(\S)*', target)

                relation_info.append(
                    f"INSERT INTO rel_btw_recipe_and_ner (Recipeid, NER_id) values ({recipe_id}, {info_for_tagging[0]});##{info_for_tagging[1]}, {found.group(0)}")
                relations.append([int(recipe_id),info_for_tagging[0]])# mysql load file 함수를 사용해보기 위해 따로 저장
                # target = re.sub(f'(\S).?{info_for_tagging[1]}.?(\S)*','',target)# regex 고구마 > target 고구마, 감자, 고구마 >> , 감자,
                target = re.sub(f'{info_for_tagging[1]}','@#',target)# regex 고구마 > target 고구마, 감자, 고구마 >> , 감자,
                if re.search(f'[ㄱ-힣]+', target) is None: break

    def custom_sort(x):
        temp = re.search('\(\S+,\s?\S+;',x).group(0).split(',')
        return temp[-1]+temp[0]

    # ingr_set = set(ingr_to_add_on_set)# 중복제거
    if len(ingr_to_add_on_set) >0:
        # ingr_set = sorted(ingr_set, key=lambda query: [i.group(0) for i in re.finditer("\(([\w]|[ㄱ-힣]|[,]|[\s]|['])*\)",query)][1])
        temp = pd.DataFrame([i.split('##') for i in ingr_to_add_on_set]).groupby(by=[0]).apply(lambda x: ','.join(x[1]))
        df = pd.DataFrame({'sql':temp.index, 'ref':temp.values}).sort_values(by=['sql'], axis =0, key=lambda col: col.apply(lambda x: col.apply(lambda x: custom_sort(x))))
        df.to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_ingr_to_add_{len(ingr_to_add_on_set)}.csv',mode='w', encoding='utf8'), header=False, index=False, sep='#' )


    # rel_set = set(relation_info)# 중복제거
    if len(relation_info) >0:
        # rel_set =  sorted(rel_set, key=lambda query: [i.group(0) for i in re.finditer("\(([\w]|[ㄱ-힣]|[,]|[\s]|['])*\)",query)][1])
        temp = pd.DataFrame([i.split('##') for i in relation_info]).groupby(by=[0]).apply(lambda x: ','.join(x[1]))
        df = pd.DataFrame({'sql':temp.index, 'ref':temp.values}).sort_values(by=['sql'], axis =0, key=lambda col: col.apply(lambda x: custom_sort(x)))
        df.to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_rel_to_add_{len(relation_info)}.csv',mode='w', encoding='utf8'), header=False, index=False, sep='#' )
        # mysql load file 함수에 사용할 파일
        pd.DataFrame(relations).to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_relations_{len(relations)}.csv',mode='w', encoding='utf8'), header=False, index=False)
        # 일치하지 않는 식재료 값

        ingr_to_add = []
        ingr_to_add_simple = []
        for i in relation_info:
            first = re.sub('([\(|\)]|\s)+',' ',i.split('##')[-1])
            second = first.split(',')[0].strip()
            third = first.split(',')[-1].strip()
            forth = re.sub('(\s*@#.*)','',third).split(',')[-1]
            if second != third:
                ingr_to_add.append(first)
            if second in forth:
                ingr_to_add_simple.append(f'{second}, {forth}')

        tempdf = pd.DataFrame(set(ingr_to_add)).sort_values(by=[0], axis =0)
        tempdf.to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_custom_ingr_to_add_{len(tempdf)}.csv',mode='w', encoding='utf8'), header=False, index=False)
        tempdf1 = pd.DataFrame(set(ingr_to_add_simple)).sort_values(by=[0], axis =0)
        tempdf1.to_csv(open(f'{path}{datetime.today().strftime("%y%m%d%H")}_custom_ingr_to_add_simple_{len(tempdf1)}.csv',mode='w', encoding='utf8'), header=False, index=False)

    save_key = 'title' if target_index ==0 else 'ingredient' if target_index ==2 else 'directions'
    filename = f'{datetime.today().strftime("%y%m%d%H")}_tagged_{save_key}.csv'
    pd.DataFrame(np.asarray(tagged_data)[:,[0,-1]]).to_csv(open(f'{path}{filename}', mode='w', encoding='utf8'), header=False, index=False)
    print(f'SAVED!! ######## {path}{filename}')
    return filename

def get_tagged_data(path, file_name, words_for_tagging = None):
    # with open(f'{path}0925_ner_298452_1.csv', mode='r', encoding='utf8') as f:
    with open(f'{path}{file_name}', mode='r', encoding='utf8') as f:

        df = pd.read_csv(f, encoding='utf8')
        data = df.to_numpy()
        
        filename_tagged = tag_data(data, words_for_tagging)
        return filename_tagged

def get_BIO_data(path, data, col_type):
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
    train = modified_data[:int(len(modified_data)*0.7)]
    val = modified_data[int(len(modified_data)*0.7):int(len(modified_data)*0.85)]
    test = modified_data[int(len(modified_data)*0.85):]
    data = {'train':train, 'val':val, 'test':test}
    # with open(f'{path}{datetime.today().strftime("%y%m%d%H")}_bio.json', 'w', encoding='utf8') as f:
    #     result = pd.DataFrame(modified_data).to_json(orient="values")
    #     parsed = json.loads(result)
    #     jsonData = json.dumps(parsed, indent=4, ensure_ascii=False)
    #     f.write(jsonData)
    # print(f'SAVED!! ######## {path}{datetime.today().strftime("%y%m%d%H")}_bio.json')
    # save_key = 'title' if col_type ==0 else 'ingredient' if col_type ==2 else 'directions'
    save_key = col_type 
    for k, v in data.items():
        file_name_to_save = f'{datetime.today().strftime("%y%m%d%H")}_bio_{save_key}_{k}.tsv'
        with open(f'{path}{file_name_to_save}', mode='a', encoding='utf8') as f:
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
        print(f'SAVED!! ######## {path}{file_name_to_save}')

# path = f'{os.path.dirname(__file__)}/data/'
# get_tagged_data(path, 'beforeTagged_2110041641_1000.csv')
sql = RecipeWithMySqlPipeline()
# data = sql.data_to_tag(5)
data = sql.data_to_tag()
wordSet = sql.words_for_tagging()
path = f'{os.path.dirname(__file__)}/data/'
totaldata = {}
for i in [2]:#0: title, 1: url 2: ingredient, 3: directions
    filename_tagged = tag_data(np.asarray(data), np.asarray(wordSet),i)# 0: title, 1: url 2: ingredient, 3: directions
    # 수작업을
    df_to_bio = pd.read_csv(f'{path}{filename_tagged}', encoding='utf8')
    # totaldata.append(df_to_bio.to_numpy())
    totaldata.update({i :df_to_bio.to_numpy()})
    # get_BIO_data(path, data, i)#f'{path}{datetime.today().strftime("%y%m%d%H%M")}_bio.tsv'

def func(param):# A,B,C >> A,B,C,AB,AC,BC,ABC
    if len(param)>1:
        result = []
        for i in func(param[1:]):
            for j in [param[:1],[]]:
                result.append(i + j)
        return result
    elif(len(param)==1):
        return param, []

# # for indexes in func([0, 2,3]):
# for indexes in func([2]):
#     an_array = None
#     for idx in indexes:
#         if isinstance(an_array,np.ndarray):
#             an_array = np.append(an_array, totaldata[idx], 0)
#         else:
#             an_array = totaldata[idx]
#     print(an_array.shape if isinstance(an_array,np.ndarray) else indexes)
#     if isinstance(an_array,np.ndarray):
#         types = '_'.join(['title' if i ==0 else 'ingredient' if i ==2 else 'directions' for i in indexes])
#         get_BIO_data(path, an_array, types)#f'{path}{datetime.today().strftime("%y%m%d%H%M")}_bio.tsv'


