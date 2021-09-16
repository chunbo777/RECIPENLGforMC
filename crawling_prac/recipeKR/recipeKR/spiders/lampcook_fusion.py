import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import re

class LampcookFusionSpider(CrawlSpider):
    name = 'lampcook_fusion'
    allowed_domains = ['lampcook.com']
    start_urls = [f'http://www.lampcook.com/food/food_fusion_list.php?pagenum={i}' for i in range(1,538)]

    rules = (
        Rule(LinkExtractor(allow=r'/food/food_fusion_view'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('div.content_tbl_90>h1.h1_title').text.strip()

            temp = soup.select('div#div_main_content>div.step_content_box>div.padd20')
            
            
            ingredients = [j.strip() for j in re.sub('(<[ㄱ-힣]+>|\\n)','',temp[1].select_one('span').text.strip(',')).split(',') if j is not None and j.strip()!='']
            directions = [i.text.strip() for i in temp[2:-1] if i is not None and i.text.strip()!='' and re.search('^[\d]',i.text.strip()) is not None]

            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()

