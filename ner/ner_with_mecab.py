import konlpy
from konlpy.tag import Mecab, Komoran, Okt
import json
import pandas as pd
import os
from tqdm import tqdm
from io import StringIO
import traceback
# me = Mecab()
# ko = Komoran()
# okt = Okt()

# with open(f'/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/recipe_kr.json', encoding='utf8') as f:
#     jsonString = '['+f.read().replace('}{','},{')+']'
#     jsondata = json.loads(jsonString)

#     for data in tqdm(jsondata):
#         data['ner_mecab'] = list(set([tup for ingr in data['ingredients'] for tup in me.pos(ingr) if tup[-1][0] == 'N']))
#         data['ner_komoran'] = list(set([tup for ingr in data['ingredients'] for tup in ko.pos(ingr) if tup[-1][0] == 'N']))
#         data['ner_okt'] = list(set([tup for ingr in data['ingredients'] for tup in okt.pos(ingr) if tup[-1] == 'Noun']))

path = '/home/dasomoh88/RECIPENLGforMC/crawling_prac/recipeKR/data/'
# with open(f'{path}recipe_kr_ner.json', mode='w', encoding='utf8') as f:
#     json.dump(jsondata, fp=f)

with open(f'{path}recipe_kr_ner.json', mode='r', encoding='utf8') as f:
    df = pd.read_json(f)
    # df.to_csv(f'{path}recipe_kr_ner.csv',encoding='utf8')



import mysql.connector

class RecipeWithMySqlPipeline:
    def __init__(self):
        self.create_connection()
        self.create_table()
    
    def create_connection(self):
        
        nameserver = None
        if os.sys.platform =='win32':
            nameserver = 'localhost'
        elif os.sys.platform == 'linux':
            nameserver = '172.20.224.1'# /etc/resolv.conf
            

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
        sql ='''
        CREATE TABLE if not exists ner_komoran (
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
        sql ='''
        CREATE TABLE if not exists ner_okt (
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
        for k in ['title','ingredients','directions']:
            
            if isinstance(item[k], str):
                item[k] = item[k].replace('"','').replace('\\','').strip()
            else:
                item[k] = [i.replace('"','').replace('\\','').strip() for i in item[k] if isinstance(i, str)]
        try:

            self.store_db(item)
        except Exception:
            traceback.print_exc()
            print(item)
        return item

    def store_db(self, item):

        sql = f'''
        INSERT INTO Recipe (title, link)
        VALUES (
            "{item['title']}"
            ,"{item['link']}"
            );
        '''
        self.curr.execute(sql)

        # self.curr.execute('SELECT LAST_INSERT_ID()')
        recipeid = str(self.curr.lastrowid)

        for ingredient in item['ingredients']:
            sql = f'''
            INSERT INTO Ingredients (Recipeid, ingredient)
            VALUES (
                "{recipeid}"
                ,"{ingredient}"
                );
            '''
            self.curr.execute(sql)

        for direction in item['directions']:
            sql = f'''
            INSERT INTO directions (Recipeid, direction)
            VALUES (
                "{recipeid}"
                ,"{direction}"
                );
            '''
            self.curr.execute(sql)
        
        for ner_mecab in item['ner_mecab']:
            sql = f'''
            INSERT INTO ner_mecab (Recipeid, ner, pos)
            VALUES (
                "{recipeid}"
                ,"{ner_mecab[0]}"
                ,"{ner_mecab[1]}"
                );
            '''
            self.curr.execute(sql)

        for ner_komoran in item['ner_komoran']:
            sql = f'''
            INSERT INTO ner_komoran (Recipeid, ner, pos)
            VALUES (
                "{recipeid}"
                ,"{ner_komoran[0]}"
                ,"{ner_komoran[1]}"
                );
            '''
            self.curr.execute(sql)

        for ner_okt in item['ner_okt']:
            if ner_okt[1] in ['Number']:
                continue
            sql = f'''
            INSERT INTO ner_okt (Recipeid, ner, pos)
            VALUES (
                "{recipeid}"
                ,"{ner_okt[0]}"
                ,"{ner_okt[1]}"
                );
            '''
            self.curr.execute(sql)

        self.conn.commit()

sql = RecipeWithMySqlPipeline()
df.apply(lambda x: sql.process_item(x), axis=1)
    
