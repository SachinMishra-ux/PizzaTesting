import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import base64
import uuid
from collections import defaultdict
import transformers
from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification


st.set_page_config(
    page_title="Named Entity Recognition Tagger", page_icon="??"
)

def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')

def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    #st.json(json_string, expanded=True)
    return json_string

st.title("Pizza NER and Intent")


######### App-related functions #########
@st.cache_resource
def load_ner_model():
    MODEL_NAME = "Ratansingh648/pizza-ner"
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


@st.cache_resource
def load_intent_model():
    MODEL_NAME = "sachin19566/intent-recog"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


id2tag = {0: 'B-Crust',
 1: 'B-Quantity',
 2: 'B-PT',
 3: 'I-Quantity',
 4: 'I-PT',
 5: 'O',
 6: 'I-Size',
 7: 'B-DT',
 8: 'B-Num',
 9: 'B-Not',
 10: 'I-Crust',
 11: 'I-Top',
 12: 'I-Num',
 13: 'I-DT',
 14: 'B-Cancel',
 15: 'B-Replace',
 16: 'B-Size',
 17: 'B-Top',
 18: 'I-Not',
 19: 'B-SPLIT'}



id2order =  { 0 : "PAYMENT",
             1 : "ORDER",
             2 : "CANCEL/MODIFY",
             3 : "INFO"}
ner_model, ner_tokenizer = load_ner_model()  
intent_model, intent_tokenizer = load_intent_model()


from dataclasses import dataclass
from typing import List

@dataclass
class Pizza:
    number : str
    pizza_type : str
    include_toppings : List[str]
    exclude_toppings : List[str]
    crust_type: str

    def __init__(self, Num=None, PT=None, Top=[], Ex_Top=[], CT=None):
        self.number = Num[0] if Num is not None else None
        self.pizza_type = PT[0] if PT is not None else None
        self.include_toppings = Top
        self.exclude_toppings = Ex_Top
        self.crust_type = CT[0] if CT is not None else None
    
    def to_json(self):
        return self.__dict__

@dataclass
class Drink:
    number: str
    drink_type: str
    size: str

    def __init__(self, Num=None, DT=None, Size=None):
        self.number = Num[0] if Num is not None else None
        self.drink_type = DT[0] if DT is not None else None
        self.size=Size[0] if Size is not None else None
    
    def to_json(self):
        return self.__dict__
    

def order_processor(text, tags):
    text = text.split()
    tags = tags.split()

    variable = None
    value = None
    entity_flag = False
    order = []
    parameter_dict = defaultdict(list)

    for index, tag in enumerate(tags):

        if tag == "B-SPLIT":
            # Append the current entity
            if entity_flag:
                parameter_dict[variable].append(value)
                entity_flag = False
            # complete current item and append to order
            if len(parameter_dict) > 0:
                if "PT" in dict(parameter_dict) or "Top" in dict(parameter_dict):
                    item = Pizza(**dict(parameter_dict))
                else:
                    item = Drink(**dict(parameter_dict))
                order.append(item.to_json())
            parameter_dict = defaultdict(list)
            # Reset variables
            variable = None
            value = None


        # if the tag is B-Tag
        elif tag.startswith("B"):
            if entity_flag:
                parameter_dict[variable].append(value)
            value = text[index]
            variable = tag.split("-")[1]
            entity_flag = True
        
        # We already know the variable type 
        elif tag.startswith("I"):
            value += " "+text[index]
        
        else:
            entity_flag = False
            if value is not None:
                parameter_dict[variable].append(value)
                variable = None
                value = None
        
    if variable is not None:
        parameter_dict[variable].append(value)
    
    if "PT" in dict(parameter_dict) or "Top" in dict(parameter_dict):
        item = Pizza(**dict(parameter_dict))
    else:
        item = Drink(**dict(parameter_dict))
    order.append(item.to_json())
    
    return order


def tagger(tokens, predictions):
    token_list = []
    tag_list = []

    print(tokens)
    print(predictions)

    prev_word = None
    for token, prediction in zip(tokens, predictions):
        # skip special tokens
        if token in ["<s>", "</s>"]:
            continue

        # include punctuations
        if token in ["."]:
            token_list.append(token)
            tag_list.append(prediction)

        # if word starts with _ its a new word
        elif token.startswith("▁"):
            token_list.append(token[1:])
            tag_list.append(prediction)

        # its part of previous word
        else:
            token_list[-1] = token_list[-1]+token
    
    return " ".join(token_list), " ".join(tag_list)



def tag_sentence(text:str):
      tokens = ner_tokenizer(text).tokens()
      inputs = ner_tokenizer(text, truncation=True, return_tensors="pt")
      outputs = ner_model(**inputs)
      probs = outputs[0][0].softmax(axis=1)
      
      word_tags = [(ner_tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()], np.round(probs[i][tagid].item() *100,2) ) 
                    for i, tagid in enumerate (probs.argmax(axis=1))]
      df=pd.DataFrame(word_tags, columns=['word', 'tag', 'probability'])
      df["tokens"] = tokens
      return df



def get_intent(text: str):
      inputs = intent_tokenizer(text, truncation=True, return_tensors="pt")
      outputs = intent_model(**inputs)
      probs = outputs[0][0].softmax(axis=-1)
      print(probs)
      print(outputs[0][0])
      label = np.argmax(probs.detach().numpy())
      return label


with st.form(key='my_form'):
    x1 = st.text_input(label='Enter a sentence:', max_chars=250)
    submit_button = st.form_submit_button(label='Process Text')



if submit_button:
    if re.sub('\s+','',x1)=='':
        st.error('Please enter a non-empty sentence.')

    elif re.match(r'\A\s*\w+\s*\Z', x1):
        st.error("Please enter a sentence with at least one word")
    
    else:
        results=tag_sentence(x1)
        labels=get_intent(x1)

        t, p = tagger(list(results["tokens"]), list(results["tag"]))
        response = order_processor(t,p)

        st.markdown("### Tagged Sentence")
        st.text("Intent : {} | {}".format(id2order[labels], labels))

        cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])
        
        c1, c2, c3 = st.columns([1, 3, 1])

        with c2:
             st.table(results.style.background_gradient(subset=['probability']).format(precision=2))

        st.markdown("### Processed JSON")
        st.json(response)

st.header("")
st.header("")
st.header("")