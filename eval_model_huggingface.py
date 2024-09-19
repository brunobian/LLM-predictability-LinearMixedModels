import re

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast 
import numpy as np

import cuentos
import current_models as cm
import paths
from tqdm import tqdm


def clean_word(word: str):
    word = word.replace("\n","")    
    chars_to_clean = "',.%-:;()\""
    for c in chars_to_clean:
        word = word.replace(c,"")
    return ' ' + word # las palabras que quiero predecir tienen que tener un espacio adelante


#def evaluate(ll: LanguageLearner, text: str):
def evaluate(model, tokenizer, text: str):
    
    words = text.split()
    target_ids = [tokenizer.encode(clean_word(target_word)) for target_word in words]
    target_ids = [x[1:] for x in target_ids]

    read_text = words[0] + " "
    probs = [(words[0], np.nan)]
    for i in tqdm(range(len(words)-1)):
        w = words[i]
        this_target = target_ids[i+1]

        if tokenizer.decode(this_target[0]) == "":
            this_target = this_target[1:]

        target_prob = 1
        for token in this_target:
            inputs = tokenizer.encode(read_text.strip(), return_tensors="pt").to('cuda')
            output = model(inputs)
            model_output = output[0]
            last_token_prediction = model_output[:, -1]
            last_token_softmax = torch.softmax(last_token_prediction, dim=-1).squeeze()

            target_prob *= last_token_softmax.tolist()[token]
            read_text += tokenizer.decode(token) 
        
        read_text += " "
        probs.append((clean_word(words[i+1]), target_prob))
        
        if i>300:
            read_text = read_text.split(" ", 1)[1]

    return probs


def main(mn):
    #tokenizer = GPT2Tokenizer.from_pretrained(paths.models / mn)
    print(paths.models / mn)

    model = AutoModelForCausalLM.from_pretrained(paths.models / mn,
                                                    device_map="auto",
                                                    load_in_8bit=True)
    tokenizer = LlamaTokenizerFast.from_pretrained(paths.models / mn)

    for c in cuentos.todos:

        print(f"predicting {c.name}")

        content = c.read()
        res = evaluate(model, tokenizer, content)
        with (c.results(mn)).open('w') as f:
            f.write('word,prob\n')
            for word, prob in res:
                f.write(f'{word},{prob}\n')



if __name__ == '__main__':
    # main(mn=cm.fine_tuned)
    # main(mn=cm.default)
    # main(mn=cm.default_maj)
    # main(mn=cm.fine_tuned_maj)
    # main(mn=cm.transformer)
    # main(mn=cm.test)
    # main(mn=cm.gpt2_ft)
    #main(mn=cm.gpt2_blogs)
    main(mn=cm.llama)


