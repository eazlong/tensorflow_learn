#!/usr/bin/python
#!coding=UTF-8
from flask import Flask, request
from xml.etree import ElementTree as ET
import data_prepare as ap


app = Flask(__name__)

predictors = {}

@app.route('/determine', methods=['POST'])
def determine():
    category = request.form['category']
    datas = request.form['datas']
    print( category, datas )

    result = datas.split('|')
    datas=datas.replace( '|', '\r\n' )
    with open( "./data/test.data", 'w' ) as file:
    	file.write(datas)

    pd = predictors[category]
    print( pd )
    p = ap.determine(pd[0], pd[1])
    i = p.index(min(p))

    return result[i]


# class request:
#     def __init__(self):
#         self.render = web.template.render()

#     def POST( self ):
#         in_data = web.input()
#         print( in_data )
#         if in_data.has_key('determine'):
#             data = web.data()
#             str_xml = web.data()
#             xml = ET.fromstring(str_xml)
#             category=xml.find("category").text
#             data=xml.find("data").text
#             ap.determine(predictors[category][0], predictors[category][1])
#                 #return self.render.reply_text(fromUser,toUser,int(time.time()), ret_msg.encode('utf-8') )
#         return invalid_request

if __name__  == '__main__':
	categorys=[["meishi.txt", "meishi"]]
	for c in categorys:
		print( c )
		predictors[c[1]] = [ap.start_train(c[1], c[0]),c[0]]

	app.run(port=8009, host="0.0.0.0") 