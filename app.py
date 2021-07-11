import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# importing data analysis elements
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import time
import requests
import math
import json
nltk.download('punkt')
st.markdown("** DISCLAIMER : No data will be captured by this app and no machine learning algorithm was harmed during this process :)**")
st.title('Optimize your resume')
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.subheader(" Please read below before submitting the Job description and the Resume ")
st.markdown("------------------------------------------------------------------------------------------------------------")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Progress bar

def status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Mr. NLTK is scanning the document,please sit back and relax : {i+1} %')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

def statusR():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Scoring your resume : {i+1} %')
		bar.progress(i + 1)
		time.sleep(0.08)
		st.empty()

def statusSal():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Estimating expected compensation : {i+1} %')
		bar.progress(i + 1)
		time.sleep(0.08)
		st.empty()
st.sidebar.subheader('Job Description\n\n')


# Sidebar Options
st.write("Copy and paste the Job Description (only the requirements and responsibilities, not 'about the company' or compensation details). For better results, paste atleast 10 similar JDs to increase the sample volume to extract the top keywords. For example, if you're applying for a Data Scientist position, paste only those JDs. Do not paste vague or general JDs. The more the presence of hard skills(technical) and soft skills in the data, the better will be the results.\n")
st.subheader(" Note that we are searching for an exact match of keywords, that's how ATS works. So, for example, 'analytics' is not equal to 'analysis', 'reporting' is not equal to 'reports'.")
st.text('\n\n')
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.subheader(" Three functions are available for you to use : ")
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.write(" 1. Get top keywords from the JD")
st.write(" 2. Get your resume score")
st.write(" 3. Predict the expected salary")
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.markdown(" 1. **Top keywords are extracted using frequency after tagging using average-perceptron-tagger. A wordcloud will be displayed based on the top keywords**  ")
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.markdown("**2. Resume score is calculated based on the frequency of occurance of keywords in JD that are present in the resume.The resume and JD differ hugely in the number of words, so the volume has been scaled down to get a reasonable score. Mutliplied by Bonus/Penalty factor for presence and absence of top keywords in resume respectively**")
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.subheader("3. Predict the expected salary")
st.markdown("**------------------------------------------------------------------------------------------------------------**")
st.write("Parameters taken into account are: ")
st.markdown("1. **Years of experience** ")
st.text("     --> Multiplied by a factor as experience increases and then flattens after a point")
st.markdown("2. **Resume match score, which is already calculated** ")
st.text("     --> Strongest factor in the calculation of expected salary")
st.markdown("3. **Academic strength and relevance** ")
st.text("     --> How relevant is your degree and your internships to the job you're applying and how strong is your academic background ?")
st.text("     --> 5 being the most strong+relevant, 4 being relevant but not strong , 3 being both average , 2 low in one, average in another, 1 poor in both")
st.text("     --> This parameter holds more weightage towards the beginning of the career, after 2 years of experience its weightage reduces")
st.write('4. **Luck factor** ')
st.text("     --> Give 0 if you do not believe in luck, 5 being extremely lucky ")

st.subheader(" Please click the checkbox if you want to get salary prediction, otherwise skip it and proceed to analysis")

#Accepting values from the user
text=st.sidebar.text_area('Paste the job description here')
st.sidebar.subheader(" Updated Resume ")
resume=st.sidebar.text_area(" Paste your resume/CV here")
st.sidebar.subheader(" Salary Prediction")

experience=st.sidebar.selectbox('Years of experience',options=(0,1,2,3,4,5,6,7,8,9,10))
relevance = st.sidebar.selectbox('Academic strength and relevance',options=(1,2,3,4,5))
luck=st.sidebar.selectbox('How lucky do you think you are ?',options=(0,1,2,3,4,5))
sal=st.sidebar.checkbox(" Predict the expected salary")

btn = st.sidebar.button("Run analysis")
proceed=False
if btn:
    st.markdown("**------------------------------------------------------------------------------------------------------------**")
    status()
    text=text.lower().strip()
  

    nltk.download('punkt')
    tokenized_sentence=sent_tokenize(text)
    tokenized_word= word_tokenize(text)
    nltk.download('stopwords')
    stop_words=set(stopwords.words("english"))
    stop_words.update(['years','experience','he/she','responsibilities','working','good','strong','preferred','like','etc','experience','\uf0b7','Mr.','Ms',',','.','(',')','&','@','#','&','*',';',':','●','·','india','India',',','indian','experience','user','work','every','requirements','qualifications','using','–',',','’','—','•','/','%'])

    filtered_sentence=[]
    for word in tokenized_word:
        if word.lower() not in stop_words:
            filtered_sentence.append(word)

    filtered_sentence=[x.lower() for x in filtered_sentence]
    nltk.download('averaged_perceptron_tagger')
    tagged_words=nltk.pos_tag(filtered_sentence)

    f_pos=[]
    for word in tagged_words:
        if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP' or word[1]=='NNPS' or word[1]=='JJ' or word[1]=='JJS' or word[1]=='JJR':
            f_pos.append(word)

    
    fdist_pos=nltk.FreqDist(filtered_sentence)
    top_100=fdist_pos.most_common(100)
    top_jd=list([x for x in top_100])
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12,8))
    st.subheader(" Top 30 keywords extracted from the job description\n")
    st.markdown("**------------------------------------------------------------------------------------------------------------**")
    plt.title(" Top 30 Keywords vs Count",fontweight='bold')
    fdist_pos.plot(30,cumulative=False)
    st.pyplot()


    new_text=' '
    for word in filtered_sentence:
        new_text=new_text+word+' '
    
    st.subheader(" Wordcloud of collected keywords")
    st.markdown("**------------------------------------------------------------------------------------------------------------**")
    st.text('\n')
    st.write(" Consider adding combinations of these words to improve hits. The size of the words imply its frequency in the JD")
    cloud=WordCloud(width=800, height=400).generate(new_text)
    plt.figure(figsize=(16,10))
    plt.imshow(cloud,interpolation='bilinear')
    plt.axis("off")
    st.pyplot()
    



    
    
    statusR()
    resume=resume.lower().strip()
    nltk.download('punkt')
    tokenized_sentence=sent_tokenize(resume)
    tokenized_word= word_tokenize(resume)
    nltk.download('stopwords')
    stop_words=set(stopwords.words("english"))
    stop_words.update(['working','good','strong','preferred','like','etc','experience','\uf0b7','Mr.','Ms',',','.','(',')','&','@','#','&','*',';',':','●','·','india','India',',','indian','experience','user','work','every','requirements','qualifications','using','–',',','’','—','•','/','%'])

    filtered_sentence=[]
    for word in tokenized_word:
        if word.lower() not in stop_words:
            filtered_sentence.append(word)

    filtered_sentence=[x.lower() for x in filtered_sentence]
    nltk.download('averaged_perceptron_tagger')
    tagged_words=nltk.pos_tag(filtered_sentence)

    f_pos=[]
    for word in tagged_words:
        if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP' or word[1]=='NNPS' or word[1]=='JJ' or word[1]=='JJS' or word[1]=='JJR':
            f_pos.append(word)
    fdist_pos=nltk.FreqDist(filtered_sentence)
    top_100=fdist_pos.most_common(100)
    top_cv=list([x for x in top_100])


    cv_dict=dict(top_cv)
    jd_dict=dict(top_jd)
    

    score=0
    missing=[]
    for word in [top_cv[x][0] for x in range(len(top_cv))]:
        if word in [top_jd[x][0] for x in range(len(top_jd))]:
            factor=float(cv_dict.get(word)/math.sqrt(jd_dict.get(word)))
            score+=(factor*cv_dict.get(word))

    for word in [top_jd[x][0] for x in range(len(top_jd[:20]))]:
        if word not in [top_cv[x][0] for x in range(len(top_cv))]:
            missing.append(word)

        


    match=float(round((score),2))
    st.markdown("**------------------------------------------------------------------------------------------------------------**")
    st.subheader(f" Your Resume match score is : {match} ")
    st.markdown("**------------------------------------------------------------------------------------------------------------**")
    st.write("What does this mean?")
    st.text("Score of 60+ --> excellent")
    st.text("Score of 40-60 --> Good")
    st.text("Score of 30-40 --> Average")
    st.text("Below 30--> poor")
    st.subheader(f" Try adding these keywords into your resume:\n {missing}  ")
    st.text("\n\n\n")
    proceed=True

    relevance_dict={5:0.95 , 4: 0.9 , 3: 0.8, 2: 0.65, 1: 0.5}
    if proceed:
        
        if experience<3:
                exp=(experience+1)*1
        else:
                exp=(1.3*experience)+1
        
        if experience<1:
                relevance=relevance_dict[relevance]*1.5
        if experience==1:
                relevance=relevance_dict[relevance]*1.35
        
        if experience>=2:
                relevance=relevance_dict[relevance]*1.05
        
        salary=round((exp*(match/10)*relevance+luck*0.4),2)
        
        
        
        if sal:
            statusSal()
            st.subheader(f"Your expected salary is : INR {salary} LPA")
            st.markdown("**------------------------------------------------------------------------------------------------------------**")
    st.text("\n\n")
    st.text(" Refresh to run the app with new inputs, re-run to try it out, also please change the theme in the streamlit settings to the top right if you wish :) ")       
    st.text("\n\n")
    st.text("App version 1.1")       
    st.markdown(" Feel free to connect with me : linkedin.com/in/abhishekiitm | abhishekvpta@gmail.com")