import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import re
import json

class ChungjungoneSpider(scrapy.Spider):
    name = 'chungjungone'
    allowed_domains = ['chungjungone.com']
    start_urls = [f'https://www.chungjungone.com/knowhow/recipe/recipeListNew.do?page={i}' for i in range(1,39)]

    # rules = (
    #     Rule(LinkExtractor(allow=r'/recipe/recipeView1New.do'), callback='parse_item', follow=True),
    # )

    def parse(self, response, **kwargs):
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            
            for i in soup.select('ul.recipe_ul.fourList>li>a'):
                temp = [i.strip('\'') for i in re.findall(r"'[\w.]+'",i.attrs['onclick'])]
                url = 'https://www.chungjungone.com/knowhow/recipe/{0}?{1}={2}'.format(*temp)
                yield response.follow(url, self.parse_item)
        except:
            traceback.print_exc()

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")
            jsonData = json.loads(re.search(r'\{[\w\W]+\}', soup.select_one('script[type="application/ld+json"]').string).group().replace('\r',''))
            
            title = jsonData['name']
            ingredients = jsonData['recipeIngredient']
            directions = [i['text'] for i in jsonData['recipeInstructions']]
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except:
            traceback.print_exc()
