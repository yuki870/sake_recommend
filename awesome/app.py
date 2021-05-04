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

#######分類器の準備
cur_dir=os.path.dirname(__file__)
lda=pickle.load(open(os.path.join(cur_dir,'pkl_objects','lda.pkl'),
                     'rb'))
mecab_dictionary=pickle.load(open(os.path.join(cur_dir,'pkl_objects','dictionary.pkl'),
                     'rb'))
#クラスタリング済みのcsvファイル読込
p = pathlib.Path('C:\\Users\\nakazawayuki\\MASSH\\sake_recommend_system\\sakeclassifier\\topic.csv')
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
    seek=[]
    sake=[]
    review=[]
    seek.append(str(text))
    seek_vec = mecab_dictionary.doc2bow(mecab_analysis(seek)[0]) #入力文章のベクトル化
    #入力情報をクラスタリング -> トピック番号の取得
    tmp=lda[seek_vec]
    max=tmp[0][1]
    topic_num=tmp[0][0]
    for j in range(len(tmp)):
        if max<tmp[j][1]:
            topic_num=tmp[j][0]
    topic_df=mecab_topic_number_df[mecab_topic_number_df['topic number'].isin([topic_num])]
    wakati_ary=topic_df['wakati'].tolist()
    simil_ary=[]
    for token in wakati_ary:
        simil_ary.append(token.split())

    #コーパスの作成
    #doc2bowで文書をbag-of-words形式に変換
    simil_corpus = [mecab_dictionary.doc2bow(text) for text in simil_ary]
    index = similarities.MatrixSimilarity(lda[simil_corpus])
    vec_lda = lda[seek_vec]
    sims=[]
    sims = index[vec_lda]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    display_name=topic_df['name'].values.tolist()
    display_review=topic_df['review'].values.tolist()
    if len(display_name)<3:
        for i in range(len(display_name)):
            sake.append(display_name[int(sims[i][0])])
            review.append(display_review[int(sims[i][0])])
    else:
        for i in range(3):
            sake.append(display_name[int(sims[i][0])])
            review.append(display_review[int(sims[i][0])])
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