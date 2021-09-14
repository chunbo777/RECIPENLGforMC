import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import json

class PhilipsSpider(CrawlSpider):
    name = 'philips'
    allowed_domains = ['philips.co.kr']
    start_urls = [f'https://www.philips.co.kr/c-m-ho/philips-chef/recipe-overview-page/_jcr_content/par/gc04v2_gridcontainer/containerpar_item_1/containerpar/n19_categorizedlisto.cards.section.(foundation---customkey---general---customkey---n19---customkey---allcategories).page.({i})' for i in range(1,15)]

    rules = (
        Rule(LinkExtractor(allow=r'/c-m-ho/philips-chef/recipe-overview-page/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")
            jsonData = json.loads(soup.select_one('script[type="application/ld+json"]').string.strip())

            # 식재료와 조리법을 출력해내는 일반화된 형식을 찾는중
            temp = soup.select_one('h1 span.p-heading-01-large')
            title = temp.text.strip() if temp.string is None else temp.string.strip()
            ingredients = jsonData['recipeIngredient'] if 'recipeIngredient' in jsonData.keys() else None
            directions = [i['text'] for i in jsonData['recipeInstructions']] if 'recipeInstructions' in jsonData.keys() else None
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except:
            traceback.print_exc()
