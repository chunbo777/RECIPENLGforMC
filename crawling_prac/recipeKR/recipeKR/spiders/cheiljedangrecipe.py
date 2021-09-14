import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import json
import re

class CheiljedangrecipeSpider(scrapy.Spider):
    name = 'cheiljedangrecipe'
    allowed_domains = ['cj.co.kr']

    def start_requests(self):
        for i in range(1,195):
            my_data = {'index': {str(i)}, 'offset': {str((i-1)*8)}}
            url = 'https://www.cj.co.kr/kr/proxy/k-food-life/cj-the-kitchen/recipe/addMore'
            yield scrapy.FormRequest(url=url, method='POST', formdata=my_data, callback=self.parse)


    def parse(self, response, **kwargs):
        try:
            jsonData = json.loads(response.text)
            
            for i in jsonData['recipeStorys']:
                url = 'https://www.cj.co.kr/kr/k-food-life/cj-the-kitchen/recipe/{}'.format(i['rSeq'])
                yield response.follow(url, self.parse_item)
        except:
            traceback.print_exc()


    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            temp = soup.select_one('div.recipe-caption>h3.title')
            title =  temp.text.strip() if temp.string is None else temp.string.strip() 
            ingredients = re.sub(r'[\\r|\\t|\\n|\\xa0]|[\s]{2}','', soup.select_one('div.ingredients-toggle>dl.first>dd>p').string).split(',')
            directions = [re.sub(r'[\\r|\\t|\\n|\\xa0]|[\s]{2}','',i.contents[-1])  for i in soup.select('div.holder>ul.order-list>li.inview-el>p.text')]
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except:
            traceback.print_exc()
