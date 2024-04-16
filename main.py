import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

import cv2
import easyocr

import pandas as pd

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

input           = "app/data/receipt-template-en-neat-750px.png"
out_ocr         = "app/data/output.txt"

def ocr(input, out_ocr):
    img = cv2.imread(input)

    # instance text detector
    reader = easyocr.Reader(['en'], gpu=True)

    # detect text on image
    text_ = reader.readtext(img)
    text_from_image=[]

    for i in range(0,len(text_)):
        text_from_image.append(text_[i][1])

    with open(out_ocr, "w") as txt_file:
        for line in text_from_image:
            txt_file.write(line + "\n")

#template setup
template = """
Answer the question based on the context below.
If you can't answer the question, replay "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
#setup openai model
model = ChatOpenAI(openai_api_key=OPEN_API_KEY, model="gpt-3.5-turbo")
#setup parser
parser = StrOutputParser()

chain = prompt | model | parser

if not os.path.exists(out_ocr):
    print ("\nNo text from image detected, proceeding to ocr ...\n")
    ocr(input,out_ocr)

print ("\nproceeding to read text file ...\n")

try:    
    with open(out_ocr) as file:
        transcription = file.read()

except Exception as err:
    print(err)

def invoker(chain, context, question):
    ai_response=chain.invoke({
        "context": context,
        "question": question

    })
    if ai_response != "I don't know.":
        if ai_response !=None:
            return ai_response
    
    else:
        invoker(chain, context, question)


# convert string to list
def convert_to_list(list_to_convert):
    list_to_convert = list_to_convert.strip('][').split(', ')
    return list_to_convert

#remove ' from list
def list_cleaner(list_to_clean):
    for i in range(0,len(list_to_clean)):
        if '' in list_to_clean[i]:
            list_to_clean[i] =list_to_clean[i].replace("'", '')

        if "" in list_to_clean[i]:
            list_to_clean[i] =list_to_clean[i].replace('"', '')
    
    return list_to_clean

try:
    ask_name = "Where is the receipt from? Output simply the name, no explanatory text, I need to store the output in a Python string variable"
    name = invoker(chain, transcription, ask_name)
    print("name: ", name)
    
    ask_date = "What date is the receipt issued? Output just the date, I need to store it in a Python string variable"
    date = invoker(chain, transcription, ask_date)
    print("date: ", date)
    
    ask_items = "What items do appear? Output them as a Python list, I need to store it in a Python list variable"
    items = invoker(chain, transcription, ask_items)
    #convert string to list
    items = convert_to_list(items)
    #remove '' from list ites
    items = list_cleaner(items)
    print("items: ", items)
    
    ask_quantity = "What is the quantity for each item? Output them as a Python list, I need to store it in a Python list variable"
    quantity = invoker(chain, transcription, ask_quantity)
    quantity=convert_to_list(quantity)
    print("quantity: ", quantity)
    
    ask_prices = "What is each item's price? Output them as a Python list, I need to store it in a Python list variable"
    items_price = invoker(chain, transcription, ask_prices)
    items_price = convert_to_list(items_price)
    items_price = list_cleaner(items_price)
    print("items_price: ", items_price)
    
    ask_subtotal = "What is the subtotal price? Output just the number, I need to store it in a Python string variable"
    subtotal = invoker(chain, transcription, ask_subtotal)
    print("subtotal: ", subtotal)

    ask_tax = "What is tax ammount? Output just the number, I need to store it in a Python variable"
    tax = invoker(chain, transcription, ask_tax)
    print("tax: ", tax)

    if items:
        dataframe_list = []
        for i in range(0,len(items)):
            if i==0:
                dataframe_list.append([items[i],quantity[i],items_price[i], subtotal, tax])
            
            else:
                dataframe_list.append([items[i],quantity[i],items_price[i]])
                
        receipt_dataframe = pd.DataFrame(dataframe_list, columns = ["items","quantity","price", "subtotal", "tax"] )
    
    #format name and date for csv writing
    formated_name = ''.join(n for n in name if n.isalnum())
    formated_date = ''.join(d for d in date if d.isalnum())
    
    csv_output      = "app/data/receipt_{}_{}.csv".format(formated_name, formated_date)

    receipt_dataframe.to_csv(csv_output)

except Exception as err:
    print(err)