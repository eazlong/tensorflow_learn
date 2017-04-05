#!/usr/bin/python
#!coding=UTF-8
from flask import Flask, request
from xml.etree import ElementTree as ET
import data_prepare as ap
import controller
import threading
#import time
import ptb_word_lm as lm

app = Flask(__name__)

predictors = {}

@app.route('/determine', methods=['POST'])
def determine():
    category = request.args.get('domain', "")
    datas = request.args.get('datas',"")
  
    result = datas.split('|')
    pd = predictors[category]
    p = controller.determine(pd[0], result, pd[2])
    i = p.index(min(p))

    return result[i]

@app.route('/train', methods=['POST'])
def train():
    category = request.form['category']
    pd = predictors[category]
    t = threading.Thread(target=controller.start_train, args=(pd[0], category, pd[1], pd[2], True))
    t.setDaemon(True)
    t.start()
    #controller.start_train(pd[0], category, pd[1], pd[2] )
    return "ok"


if __name__  == '__main__':
    categorys=[["data/catering.txt", "catering", "data/catering_vocab.txt"]]
    for c in categorys:
        p = lm.Predictor(c[1])
     
        #time.sleep(2)
        controller.start_train(p, c[1], c[0], c[2], False)
        predictors[c[1]] = [p,c[0],c[2]]
        
    app.run(port=80, host="0.0.0.0") 
