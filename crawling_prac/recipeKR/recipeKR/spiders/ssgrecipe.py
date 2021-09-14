import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback

class SsgrecipeSpider(CrawlSpider):
    name = 'ssgrecipe'
    allowed_domains = ['emart.ssg.com']

    download_delay = 1.4
    start_urls = [f'http://emart.ssg.com/recipe/list.ssg?page={i}&sort=regdt&searchType=list' for i in range(3120, 3860)]

    rules = (
        Rule(
            LinkExtractor(allow=r'/recipe/recipe/detail.ssg'), callback='parse_item', follow=True
            # , process_links='process_link', process_request='process_req'
        ),
    )

    # def process_link(self, response):
    #     print(response)#debug
    #     return response
    # def process_req(self, request, response):
    #     print(request)
    #     return request#debug

    def parse_item(self, response):
        # if '/recipes/' not in response.request.url:
        #     return
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('h2.post_subject').text.strip()
            ingredients = [i.text.strip() for i in soup.select('dl.recipe_ingredient a.btn_hash') if i is not None]
            directions = [i.text.strip() for i in soup.select('dl.recipe_step p.dsc') if i is not None]
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()
