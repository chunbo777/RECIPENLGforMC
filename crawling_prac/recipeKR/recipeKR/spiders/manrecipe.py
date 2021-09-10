from logging import exception
from os import path
import scrapy
from ..items import RecipekrItem
from bs4 import BeautifulSoup
import json
import csv
import os.path
import pandas as pd
import sys
from itertools import takewhile

class ManrecipeSpider(scrapy.Spider):
    name = 'manrecipe'
    allowed_domains = ['10000recipe.com']
    page_num = 1
    download_delay = 1.5
    urlPath = './manRecipeUrls.txt'
    urlFile = open('./second.txt', 'r', encoding='utf8')
    u=[1]
        
    def start_requests(self):

        if os.path.isfile(ManrecipeSpider.urlPath):
            url = f'https://www.10000recipe.com{ManrecipeSpider.urlFile.readline()}'
            yield scrapy.Request(url=url, callback=self.parse)

        else:
            urls = [
                f'https://www.10000recipe.com/recipe/list.html?order=date&page={ManrecipeSpider.page_num}',
            ]
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parseUrl)
            

    def parseUrl(self, response):
        if response.status == 200:
            with open(ManrecipeSpider.urlPath, 'a', encoding='utf8') as f:
                for recipeInfo in  response.css('li.common_sp_list_li').extract():
                    soup = BeautifulSoup(recipeInfo, "html.parser")
                    f.write(soup.select_one('a.common_sp_link').attrs['href']+'\n')

            # yield items
            ManrecipeSpider.page_num += 1
            next_recipe = f'https://www.10000recipe.com/recipe/list.html?order=date&page={ManrecipeSpider.page_num}'
            yield response.follow(next_recipe, callback= self.parseUrl)


    def getString(self, el):
        if isinstance(el, str):
            return el.strip()
        else:
            return self.getString(el.next_element)

    def parse(self, response):
        if response.status == 200:
            items = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser") 
            title = soup.select_one('div.view2_summary.st3>h3').next_element
            try:
                if len(title)>0:
                    ingredients = [ingredient for dd in response.css('div.cont_ingre dd::text').extract() for ingredient in dd.split(',')]
                    if len(ingredients) ==0:
                        for sel in soup.select('div#divConfirmedMaterialArea>ul>a>li'):
                            ingredient = sel.next.strip()
                            unit = sel.select_one('span.ingre_unit').next.strip()
                            ingredients.append(f'{ingredient} {unit}'.strip())

                    directions = []
                    for direction in soup.select('div.view_step_cont div.media-body'):
                        # temp = []
                        for x in direction:
                            if  isinstance(x, str) or not x.is_empty_element:
                                # temp.append(self.getString(x))
                                directions.append(self.getString(x))
                        # directions.append(' '.join(temp))
                    link = response.url

                    items['title'] = title
                    items['ingredients'] = ingredients
                    items['directions'] = directions
                    items['link'] = link
                
                    if items['ingredients'] and items['directions']:
                        yield items

                else:
                    self.logger.warning("Unable to get recipe from: " + response.url)
            except Exception as e:
                print(sys.exc_info())
                self.logger.warning("Unable to get recipe from: " + response.url)

            url =  f'https://www.10000recipe.com{ManrecipeSpider.urlFile.readline()}'
            yield scrapy.Request(url=url, callback=self.parse)
            # next_recipe = 'https://www.10000recipe.com/recipe/'+ str(int(response.request.url.split('/')[-1])+1)
            # yield response.follow(next_recipe, callback= self.parse)
        else:
            self.logger.warning("Unable to get recipe from: " + response.url)
            
            url =  f'https://www.10000recipe.com{ManrecipeSpider.urlFile.readline()}'
            yield scrapy.Request(url=url, callback=self.parse)
            # next_recipe = 'https://www.10000recipe.com/recipe/'+ str(int(response.request.url.split('/')[-1])+1)
            # yield scrapy.Request(next_recipe, callback=self.parse, dont_filter=True)
