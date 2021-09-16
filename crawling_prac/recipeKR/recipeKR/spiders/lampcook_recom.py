import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import re

class LampcookRecomSpider(CrawlSpider):
    name = 'lampcook_recom'
    allowed_domains = ['lampcook.com']
    start_urls = [f'http://www.lampcook.com/food/food_recom_list.php?big_no=0&pagenum={i}' for i in range(1,23)]

    rules = (
        Rule(LinkExtractor(allow=r'/food/food_recom_view'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            for el in soup.select('ul.glo_ul_title_sub4'):
                title = el.select('li')[-1].text.strip()

                temp = el.next_sibling.next_sibling.next_sibling.next_sibling.select_one('td.td_txt_color10').next_sibling.next_sibling

                ingr = re.sub('(<[ㄱ-힣]+>|\[[\s\S]*\]|\\n)','',temp.text.replace('\r',',').strip())
                # reg2 = re.search('\([,|ㄱ-힣]*\)',reg1)
                # ingr = re.sub('[\s]*[,]{1}[\s]*',' ',reg2.group()).join(re.split('\([,|ㄱ-힣]*\)', reg1)) if reg2 is not None else reg1
                ingredients = [j.strip() for j in ingr.split(',') if j is not None and j.strip()!='']

                dir = el.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.select_one('div.txt_padd_box20')
                
                directions = [i.strip() for i in dir.contents if i is not None and isinstance(i, str)]

                link = response.url

                item['title'] = title
                item['ingredients'] = ingredients
                item['directions'] = directions
                item['link'] = link

                yield item
        except Exception as e:
            traceback.print_exc()

