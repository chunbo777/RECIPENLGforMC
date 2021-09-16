from sklearn.model_selection import train_test_split
import pandas as pd
import re
import traceback

df = pd.read_csv("/home/lab17/RECIPENLGforMC/generation_prac/datain/full_dataset.csv")
df1 = pd.read_csv("/home/lab17/RECIPENLGforMC/generation_prac/datain/data_1m.csv", )

import re


# It is crucial to create the model that generate ”rich”, extensive recipes. 
# Therefore, we removed recipes that do not provide the model with sufficiently comprehensive information
# , such as one-ingredient recipes or recipes with short instructions. 
# Part of generating a recipe is the title generation. 
# We intend to generating the title strictly related to the content of the recipe
# df.drop(df[df.title.map(lambda x: len(x)<4)].index, inplace=True)
# df.drop(df[df.ingredients.map(lambda x: len(x)<2)].index, inplace=True)
# df.drop(df[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)].index, inplace=True)
# It is also impossible to check if the model has learned to refer to previous steps correctly.
# The incorrect use of the word ’step’ causes losing the meaning of the entire instruction
# df.drop(df[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

df.drop(df[df[df.columns[1]].map(lambda x: len(x)<4)].index, inplace=True)
df.drop(df[df[df.columns[2]].map(lambda x: len(x)<2)].index, inplace=True)
df.drop(df[df[df.columns[3]].map(lambda x: len(x) < 2 or len(''.join(x)) < 30)].index, inplace=True)
df.drop(df[df[df.columns[3]].map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

df.reset_index(drop=True, inplace=True)


import numpy as np
'''
20210902
This was the moment where 5% of evaluation data was extracted from the dataset. 
The amount of data was intentionally selected big enough to be further divided and used as dev set and test set. 
The additional test set was also prepared for the purpose of checking evaluation metrics. 

'''
# 학습데이터의 크기조정
# df = df.iloc[:int(df.shape[0]/4)]

import json
with open('./eceptions1.txt', 'w', encoding="utf-8") as f:
    for index, row in df1.iterrows():
        if index%100000==0:
            print(index)
        for i in range(3):
            try:
                json.loads(row[row.index.values[i+2]].replace('\u200b',''))
            except Exception as e:
                traceback.print_exc()
                f.write("{}\r\n{}\r\n".format(' ,'.join([e.args[-1], row[row.index.values[i+2]], row.index.values[i+2], str(index)]),df.iloc[index][df.columns[i+2]]))




train, test = train_test_split(df, test_size=0.05) #use 5% for test set
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


def df_to_plaintext_file(input_df, output_file):
    print("Writing to", output_file)
    with open(output_file, 'w', encoding="utf-8") as f:

        for index, row in input_df.iterrows():
            if index%100000==0:
                print(index)
            # if type(row.NER)!=str:
            if type(row[row.index.values[-1]])!=str:
                continue
            # title = row.title
            # directions = json.loads(row.directions)
            # ingredients = json.loads(row.ingredients)
            # ner = json.loads(row.NER)
            title = row[row.index.values[1]].replace('\u200b','')
            directions = re.split('"[,\s|\s,]+["]*', row[row.index.values[2]].replace('\u200b','').strip(' "[]"'))
            ingredients = re.split('"[,\s|\s,]+["]*', row[row.index.values[3]].replace('\u200b','').strip(' "[]"'))
            ner = re.split('"[,\s|\s,]+["]*', row[row.index.values[4]].replace('\u200b','').strip(' "[]"'))

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

# df_to_plaintext_file(train, '/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_train_kr_1m_translated_short.txt')
# df_to_plaintext_file(test, '/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_test_kr_1m_translated_short.txt')
