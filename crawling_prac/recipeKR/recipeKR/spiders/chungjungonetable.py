import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback


class ChungjungonetableSpider(CrawlSpider):
    name = 'chungjungonetable'
    allowed_domains = ['chungjungone.com']
    start_urls = [f'https://www.chungjungone.com/knowhow/table/tableListNew.do?page={i}' for i in range(1,79)]

    rules = (
        Rule(LinkExtractor(allow=r'/knowhow/table/tableViewNew.do'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")
            
            title = soup.select_one('div.editRecipy>h2').string.strip()

            # 식재료와 조리법을 출력해내는 일반화된 형식을 찾는중
            ingredients = None# 아예 없는 페이지가 있음, ner을 사용해야 할것 같습니다.
            directions = [i.string.strip() for i in soup.select('div.wrap1100>div.editor-wrap>p') if i.string is not None and i.string.strip()!='']
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            # yield item
        except:
            traceback.print_exc()
