from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/home/lab17/recipe_generation/recipenlg/generation/datain/full_dataset.csv")

import re
df.drop(df[df.title.map(lambda x: len(x)<4)].index, inplace=True)
df.drop(df[df.ingredients.map(lambda x: len(x)<2)].index, inplace=True)
df.drop(df[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)].index, inplace=True)
df.drop(df[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

df.reset_index(drop=True, inplace=True)


import numpy as np
'''
20210902
This was the moment where 5% of evaluation data was extracted from the dataset. 
The amount of data was intentionally selected big enough to be further divided and used as dev set and test set. 
The additional test set was also prepared for the purpose of checking evaluation metrics. 

'''
train, test = train_test_split(df, test_size=0.05) #use 5% for test set
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

import json

def df_to_plaintext_file(input_df, output_file):
    print("Writing to", output_file)
    with open(output_file, 'w', encoding="utf-8") as f:
        for index, row in input_df.iterrows():
            if index%100000==0:
                print(index)
                if index > 0:#20210830 학습시간을 줄이기 위해 생성
                    break
            if type(row.NER)!=str:
                continue
            title = row.title
            directions = json.loads(row.directions)
            ingredients = json.loads(row.ingredients)
            ner = json.loads(row.NER)
            '''
            20210902
            Control tokens were inserted into the recipe. 
            Each recipe was embraced with RECIPE tokens and consisted of: 
                list of inputs, list of ingredients, list of instructions, and finally: a title. 
            The order of elements of recipe was made for purpose. 
            It was decided, that for the generative left-to-right model
                , it is reasonable to first provide it with input, context data
                , then allow it to generate the recipe itself
                , concluding with the title with regard to all the generated data.
            '''
            res = "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(ner) + " <INPUT_END> <INGR_START> " + \
              " <NEXT_INGR> ".join(ingredients) + " <INGR_END> <INSTR_START> " + \
              " <NEXT_INSTR> ".join(directions) + " <INSTR_END> <TITLE_START> " + title + " <TITLE_END> <RECIPE_END>"
            f.write("{}\n".format(res))

df_to_plaintext_file(train, '/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised_train_short.txt')
df_to_plaintext_file(test, '/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised_test_short.txt')