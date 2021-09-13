import scrapy
import json
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback

class WtableSpider(scrapy.Spider):
    name = 'wtable'
    allowed_domains = ['wtable.co.kr']
    # start_urls = ['http://wtable.co.kr/recipes']
    
    with open('./data/wtableRecipeToken.json', encoding='utf8') as f:
        jsonData = json.load(fp=f)['data']
        start_urls = [f'https://wtable.co.kr/recipes/{data["token"]}' for data in jsonData]

    def parse(self, response, **kwargs):
        try:

            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('h1.style__Title-xl9np2-4.EchoF').text.strip()
            ingredients = [ ' '.join([j.text.strip() for j in i.select('div') if j is not None and j.text.strip()!='']).strip() for i in soup.select('div.ingredient>ul.igroups>li>ul>li>div')]
            directions = [i.text.strip() for i in soup.select('section.Section-sc-1czqlgp-0.hnJtAl>div.steps>div.token__Step-sc-1o2h3sm-1.ihCzrN>div>p') if i is not None and i.text.strip()!='']
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()
