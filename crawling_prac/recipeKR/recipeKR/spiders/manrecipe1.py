import scrapy
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback

class Manrecipe1Spider(scrapy.Spider):
    name = 'manrecipe1'
    allowed_domains = ['10000recipe.com']
    start_urls = ['http://10000recipe.com/']

    with open('./second.txt', 'r', encoding='utf8') as f:
        start_urls = [f'https://www.10000recipe.com{line}' for line in f.readlines()]

    def parse(self, response, **kwargs):
        try:

            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('div.view2_summary.st3>h3').text.strip()
            ingredients = [ ' '.join([i.contents[0].strip(), i.select_one('span.ingre_unit').text.strip()]).strip() for i in soup.select('div#divConfirmedMaterialArea>ul li') if len(i.contents)>2]
            directions = []
            for i in soup.select('div.view_step_cont div.media-body'):
                if i is not None:
                    for j in i:
                        if isinstance(j, str) and j.strip()!='':
                            directions.append(j.strip())
                        elif j is not None and j.text.strip()!='':
                            directions.append(j.text.strip())

            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()
