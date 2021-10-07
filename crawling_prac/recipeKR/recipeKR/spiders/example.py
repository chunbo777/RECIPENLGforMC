import scrapy
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback
import re
import json

class ExampleSpider(scrapy.Spider):
    name = 'example'
    allowed_domains = ['www.lampcook.com']
    start_urls = ['http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=122'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=119'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=123'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=115'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=114'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=118'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=117'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=121'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=120'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=124'
,'http://www.lampcook.com/food/food_wellbeing_view.php?idx_no=116']

    def parse(self, response, **kwargs):
        try:
            # print(f"")
            soup = BeautifulSoup(response.text, "html.parser")
            def dashrepl(matchobj):
                target = matchobj.group(0)
                if target is not None :
                    if ',' in target: 
                        return f",{target.strip('()')}"
                    elif '인분' in target or '기준' in target:
                        return ''
                    else:
                        return target
                else: 
                    return target
            text = re.sub('(\(([ㄱ-힣]|[\d]|[\s]|[\w]|[/]|[,])*\))',dashrepl,soup.select('div.step_content_box div.padd20 span')[1].text)
            ingredients = re.split('\s*[,]\s*',re.sub('(<[ㄱ-힣]+>|\\r|\\n)','',text))
            with open(f'./data/additional_sql.txt', mode='a', encoding='utf8') as f:
                f.writelines([f"insert into ingredients (Recipeid, ingredient) select max(Recipeid) Recipeid, '{ingr.strip()}' from recipe where link = '{response.url}' group by Recipeid;\n" for ingr in ingredients if ingr.strip()!=''])
        except:
            traceback.print_exc()

    # def parse_item(self, response):
    #     try:
    #         item = RecipekrItem()
    #         soup = BeautifulSoup(response.text, "html.parser")
    #         jsonData = json.loads(re.search(r'\{[\w\W]+\}', soup.select_one('script[type="application/ld+json"]').string).group().replace('\r',''))
            
    #         title = jsonData['name']
    #         ingredients = jsonData['recipeIngredient']
    #         directions = [i['text'] for i in jsonData['recipeInstructions']]
    #         link = response.url

    #         item['title'] = title
    #         item['ingredients'] = ingredients
    #         item['directions'] = directions
    #         item['link'] = link

    #         yield item
    #     except:
    #         traceback.print_exc()
