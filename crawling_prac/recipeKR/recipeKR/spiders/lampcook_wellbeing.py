import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import re

class LampcookWellbeingSpider(CrawlSpider):
    name = 'lampcook_wellbeing'
    allowed_domains = ['lampcook.com']
    start_urls = [f'http://www.lampcook.com/food/food_wellbeing_list.php?pagenum={i}' for i in range(1,137)]

    rules = (
        Rule(LinkExtractor(allow=r'/food/food_wellbeing_view'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('div.content_tbl_90>h1.h1_title').text.strip()

            temp = soup.select('div#div_main_content>div.step_content_box>div.padd20')
            
            reg1 = re.sub('(<[ㄱ-힣]+>|\[[\s\S]*\]|\\n)','',temp[0].select_one('span').text.replace('\r',',').strip(','))
            reg2 = re.search('\([,|ㄱ-힣]*\)',reg1)
            ingr = re.sub('[\s]*[,]{1}[\s]*',' ',reg2.group()).join(re.split('\([,|ㄱ-힣]*\)', reg1)) if reg2 is not None else reg1
            ingredients = [j.strip() for j in ingr.split(',') if j is not None and j.strip()!='']
            directions = [i.text.strip() for i in temp[2:-1] if i is not None and i.text.strip()!='' and re.search('^[\d]',i.text.strip()) is not None]

            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()

