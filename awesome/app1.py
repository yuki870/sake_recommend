# -*- coding: utf-8 -*-
from flask import Flask,render_template,request
from wtforms import Form,TextAreaField,validators
import pickle
import os
import re
import MeCab
import pandas as pd
from gensim import corpora, models, similarities
import pathlib

app=Flask(__name__)

#######辞書・コーパスの準備
cur_dir=os.path.dirname(__file__)
mecab_corpus=pickle.load(open(os.path.join(cur_dir,'pkl_objects1','corpus.pkl'),
                     'rb'))
mecab_dictionary=pickle.load(open(os.path.join(cur_dir,'pkl_objects1','dictionary.pkl'),
                     'rb'))
#クラスタリング済みのcsvファイル読込
p = pathlib.Path('C:\\Users\\nakazawayuki\\system\\sakeclassifier\\topic.csv')
mecab_topic_number_df=pd.read_csv(p.resolve(),header=None,encoding='CP932')
mecab_topic_number_df.drop(mecab_topic_number_df.index[[1]])
mecab_topic_number_df.columns=['name','review','wakati','topic number']
mecab_topic_number_df=mecab_topic_number_df.drop(mecab_topic_number_df.index[[0]])
mecab_topic_number_df['topic number']=mecab_topic_number_df['topic number'].astype(int)


#形態素解析を行う関数
def mecab_analysis(ary):
    #neologdの使用
    tagger = MeCab.Tagger('-Ochasen C:\\Users\nakazawayuki\mecab-ipadic-neologd\bin')
    #エラーの回避
    tagger.parse('')

    tmp=[]
    tmp1=""
    mecab_inf=[]
    for text in ary:
        node=tagger.parseToNode(text)
        while node:
            if node.feature.split(",")[0] == u"名詞":
                tmp.append(node.surface)
                tmp1=" ".join(tmp)
                node = node.next
            elif node.feature.split(",")[0] == u"形容詞":
                tmp.append(node.feature.split(",")[6])
                tmp1=" ".join(tmp)
                node = node.next
            elif node.feature.split(",")[0] == u"動詞":
                tmp.append(node.feature.split(",")[6])
                tmp1=" ".join(tmp)
                node = node.next
            else:
                node = node.next       
        mecab_inf.append(tmp1)
        tmp=[]

    word=re.compile("[０-９a-zA-Z！!%％「」、。．”’（）+,-?？@^0-9&℃♪\*\\\"\'±()__‼❗×“…※→○◎☆々〆・『』◆《》【】〒〇▼△↓÷← βαΘε‥〔￥]")
    text =re.compile('<[^>]*>')
    emotions = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
    for i in range(len(mecab_inf)):
        mecab_inf[i]=word.sub(" ",mecab_inf[i]) 
        mecab_inf[i]=text.sub(" ",mecab_inf[i])
        mecab_inf[i]=emotions.sub(" ",mecab_inf[i])

    mecab_information=[]
    for text in mecab_inf:
        mecab_information.append(text.split(" "))
    for li in mecab_information:
        for i in range(len(li)):
            if len(li[i])==1:
                   li[i]=li[i].replace(li[i],'')

    return mecab_information

def recommend(text):
    sake=[]
    review=[]
    seek=[]
    seek.append(text)
    seek_vec = mecab_dictionary.doc2bow(mecab_analysis(seek)[0]) 
    index=similarities.MatrixSimilarity(mecab_corpus,num_features=len(mecab_dictionary))
    simil=index[seek_vec]
    tuple_list=[]
    for i in range(len(simil)):
        tuple_list.append((simil[i],i))
    tuple_list=sorted(tuple_list,reverse=True)
    for i in range(3):
        sake.append(mecab_topic_number_df['name'][tuple_list[i][1]+1])  #読み込み時にインデックス番号1ずれたため
        review.append(mecab_topic_number_df['review'][tuple_list[i][1]+1])
        
    return zip(sake,review)

#######Flask
class ReviewForm(Form):
    sakereview=TextAreaField('',
                              [validators.DataRequired(),
                               validators.length(min=1)])  #入力文字を最低1文字とする
@app.route('/')
def index():
    form=ReviewForm(request.form)
    return render_template('reviewform.html',form=form)

@app.route('/results',methods=['POST'])
def results():
    form=ReviewForm(request.form)
    if request.method=='POST' and form.validate():
        review=request.form['sakereview']
        result=recommend(review)
        return render_template('results.html',
                               content=review,
                               data=result
                               )
    return render_template('reviewform.html',form=form)

@app.route('/questionnaire')
def quesionnaire():
    #form=ReviewForm(request.form)
    #if request.method=='POST' and form.validate():
      #  return render_template('questionnaire.html',form=form)
    return render_template('questionnaire.html')

@app.route('/thanks',methods=['POST'])
def thanks():
    form=ReviewForm(request.form)
    #if request.method=='POST' and form.validate():
     #   return render_template('thanks.html',form=form)
    return render_template('thanks.html',form=form)

if __name__=='__main__':
    app.run(debug=True)
