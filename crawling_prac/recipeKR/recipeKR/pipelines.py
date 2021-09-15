# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

'''
Scraped data -> Item Containers -> Json/CSV files
Scraped data -> Item Containers -> Pipeline -> SQL/MongoDB
'''

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import os
# import mysql.connector
# import pymongo
import json

# class RecipekrWithMongoPipeline(object):
#     def __init__(self):
#         self.conn = pymongo.MongoClient(
#             host='localhost', port=27017
#         )
#         db = self.conn['myquotes']
#         # self.collection = db['quotes_prac']
#         self.collection = db['recipe_prac']

#     def process_item(self, item, spider):

#         self.collection.insert(dict(item))

#         return item



# class RecipeWithMySqlPipeline:
#     def __init__(self):
#         self.create_connection()
#         self.create_table()
    
#     def create_connection(self):
        
#         nameserver = None
#         if os.sys.platform =='win32':
#             nameserver = 'localhost'
#         elif os.sys.platform == 'linux':
#             nameserver = '172.19.128.1'# /etc/resolv.conf
            

#         mydb = mysql.connector.connect(
#             charset='utf8'
#             , db='Recipe'
#             , host=nameserver
#             , user="dasomoh"
#             , password="1234"
#         )
#         self.conn = mydb
#         self.curr = self.conn.cursor(buffered=True)
    
#     def create_table(self):
        
#         sql ='''
#         CREATE TABLE if not exists Recipe (
#             Recipeid int NOT NULL AUTO_INCREMENT,
#             title varchar(255) NOT NULL,
#             link varchar(255),
#             PRIMARY KEY (Recipeid)
#         );
#         '''
#         self.curr.execute(sql)
#         sql ='''
#         CREATE TABLE if not exists Ingredients (
#             IngrId int NOT NULL AUTO_INCREMENT,
#             Recipeid int NOT NULL,
#             ingredient varchar(255),
#             PRIMARY KEY (IngrId),
#             FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
#         );
#         '''
#         self.curr.execute(sql)
#         sql ='''
#         CREATE TABLE if not exists Directions (
#             Dirid int NOT NULL AUTO_INCREMENT,
#             Recipeid int NOT NULL,
#             direction varchar(1023),
#             PRIMARY KEY (Dirid),
#             FOREIGN KEY (Recipeid) REFERENCES Recipe(Recipeid)
#         );
#         '''
#         self.curr.execute(sql)
#         # self.curr.execute("""DROP TABLE IF EXISTS recipe_prac""")

#         # self.curr.execute("""create table recipe_prac(
#         #     title text, author text, tag text
#         # )""")
    


#     def process_item(self, item, spider):

#         self.store_db(item)

#         return item

#     def store_db(self, item):
#         sql = f'''
#         INSERT INTO Recipe (title, link)
#         VALUES (
#             "{item['title'][0].strip()}"
#             ,"{item['link']}"
#             );
#         '''
#         self.curr.execute(sql)

#         self.curr.execute('SELECT LAST_INSERT_ID()')
#         recipeid = str(self.curr.lastrowid)

#         for ingredient in item['ingredients']:
#             sql = f'''
#             INSERT INTO Ingredients (Recipeid, ingredient)
#             VALUES (
#                 "{recipeid.strip()}"
#                 ,"{ingredient.strip()}"
#                 );
#             '''
#             self.curr.execute(sql)

#         for direction in item['directions']:
#             sql = f'''
#             INSERT INTO directions (Recipeid, direction)
#             VALUES (
#                 "{recipeid.strip()}"
#                 ,"{direction.strip()}"
#                 );
#             '''
#             self.curr.execute(sql)

#         self.conn.commit()


class RecipekrPipeline:
    def __init__(self):
        pass

    def process_item(self, item, spider):

        # with open('./data/wtablerecipe.json', 'a', encoding='utf8') as f:
        # with open('./data/haemukrecipe.json', 'a', encoding='utf8') as f:
        # with open('./data/ssgrecipe.json', 'a', encoding='utf8') as f:
        # with open('./data//manrecipe1.json', 'a', encoding='utf8') as f:
        # with open('./data/chungjungone.json', 'a', encoding='utf8') as f:
        # with open('./data/philips.json', 'a', encoding='utf8') as f:
        # with open('./data/cheiljedang.json', 'a', encoding='utf8') as f:
        with open('./data/lampcook_north.json', 'a', encoding='utf8') as f:

            jsonData = json.dumps(item._values, ensure_ascii=False)
            f.write(jsonData)

        return item
