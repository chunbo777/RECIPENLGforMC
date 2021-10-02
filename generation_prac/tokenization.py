from transformers import GPT2Tokenizer, PreTrainedTokenizerFast
import h5py
import numpy as np
import os

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
special_tokens = {
    "additional_special_tokens": [
        "<TITLE_START>",
        "<TITLE_END>",
        "<INSTR_START>",
        "<NEXT_INSTR>",
        "<INSTR_END>",
        "<INGR_START>",
        "<NEXT_INGR>",
        "<INGR_END>",
        "<RECIPE_START>",
        "<RECIPE_END>",
        "<INPUT_START>",
        "<INPUT_END>",
        "<NEXT_INPUT>"
    ]
}

#20210902
#The only difference was addition of recipe control tokens to the set of known tokens before tokenization started.
tokenizer.add_special_tokens(special_tokens)

end_token_id = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]

hf = h5py.File(f"{os.path.dirname(__file__)}/datain/unsupervised_translated_short.h5", "w")
for filename in ["unsupervised_test_kr_1m_translated_short", "unsupervised_train_kr_1m_translated_short"]:
    out_np = []
    data = open(f"{os.path.dirname(__file__)}/datain/"+filename+".txt", "r", encoding='utf-8')
    num = 0
    rows = 0
    last=[]
    for line in data:
        num+=1
        if num%10000 == 0:
            print("Read "+str(num)+" Written: "+str(rows))

        text_tokens = tokenizer.tokenize(line)#subword tokenizer

        # 20210902
        # In GPT-2 language model, the maximum amount of tokens processed as one context  is equal to 1024. 
        if len(text_tokens) > 1024: #Recipe won't fit the model
            continue

        text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # 20210902
        # As it was discovered that the recipes consist of much smaller amount of tokens 
        #   (around 300 tokens per recipe), it was decided that multiple training samples 
        #   may be squashed into one context window delivered as a training example.
        # This way, the training time on the whole dataset was made more than 3 times shorter.
        # While there was a risk of lowering the quality due to having contexts coming 
        #   from different recipes in the single context window, 
        #   no significant model performance decrease was observed.
        if (len(last) + len(text_tokens_ids)) <= 1024:
            last+=text_tokens_ids
        else:
            while len(last) < 1024:
                last.append(end_token_id)
            out_np.append(last)
            last=text_tokens_ids
            rows+=1
    out_mat = np.matrix(out_np)
    print(out_mat.shape)#(n_squashed, 1024) 
    '''
    # 210902
    data processing more efficient.
    fast random lookups into the memory while removing the need of loading two files
    '''
    hf.create_dataset(filename, data=out_mat)
hf.close()
