import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import base64
import uuid

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
    MODEL_NAME = "Ratansingh648/fastfood-ner"
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


@st.cache_resource
def load_intent_model():
    MODEL_NAME = "sachin19566/distilbert_Pizza_Intent"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


id2tag={0: 'B-Crust',
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
 18: 'I-Not'}

ner_model, ner_tokenizer = load_ner_model()  
intent_model, intent_tokenizer = load_intent_model()

def tag_sentence(text:str):
      inputs = ner_tokenizer(text, truncation=True, return_tensors="pt")
      outputs = ner_model(**inputs)
      probs = outputs[0][0].softmax(axis=1)
      word_tags = [(ner_tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()], np.round(probs[i][tagid].item() *100,2) ) 
                    for i, tagid in enumerate (probs.argmax(axis=1))]

      df=pd.DataFrame(word_tags, columns=['word', 'tag', 'probability'])
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

        st.markdown("### Tagged Sentence")
        st.text("Intent : {}".format(labels))

        cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])
        
        c1, c2, c3 = st.columns([1, 3, 1])

        with c2:
             st.table(results.style.background_gradient(subset=['probability']).format(precision=2))


st.header("")
st.header("")
st.header("")