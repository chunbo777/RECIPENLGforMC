import os, io
import mysql.connector
import re
class EntitiesWithMySqlPipeline:
    def __init__(self):
        self.create_connection()
    
    def create_connection(self):

        mydb = mysql.connector.connect(
            charset='utf8'
            , db='recipe'
            , host='3.37.218.4', port = '3306'# team서버에서는 연결이 안됨#Authentication plugin 'caching_sha2_password' cannot be loaded
            , user="aws_mysql_lab17"# 연결이 안됨
            , password="1234"
        )
        self.conn = mydb
        self.curr = self.conn.cursor(buffered=True)

    def get_entities(self, inputs):
        sql = f'''
        select word from words_for_tagging 
        where in_use =1 and cate ='ingr' 
        and word in ("{'","'.join([input for input in inputs if input.strip()!=''])}"); 
        '''
        self.curr.execute(sql)# query 수행
        result = self.curr.fetchall()# query 결과 추출
        return result
    

sql = EntitiesWithMySqlPipeline()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/lab17/RECIPENLGforMC/webapp/generation/exemplary-oath-326109-4b2e00f97193.json"
# Imports the Google Cloud client library
from google.cloud import vision
import re
def detect_text_uri(file):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    client = vision.ImageAnnotatorClient()
    image =  vision.Image(content=file.read())
    # image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return sql.get_entities([re.sub('(\n|"|[ㄱ-ㅎㅏ-ㅣ]|\d|,|:|-)','',text.description) for i, text in enumerate(texts) if i >0])

# detect_text_uri('http://aquair.pe.kr/wp/blog/wp-content/uploads/sites/2/2021/07/20210714_1549027_018.jpg')
