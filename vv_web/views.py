
from _csv import writer

from django.http import HttpResponse
from wsgiref.util import FileWrapper

import pickle
import os
from django.shortcuts import render
from django.template.context_processors import csrf


from django.shortcuts import render
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
from datetime import datetime
sentiment_model = tf.keras.models.load_model(r'C:\Users\jeeva\Downloads\modelfinal.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def home(request):
    return render(request, 'hom.html')
def inx(request):
    return render(request,'index.html')

file_path1='vv_web/datac.csv'
file_path2='vv_web/datap.csv'
file_path3='vv_web/dataj.csv'
def result(request):
    Teaching=request.GET['Teaching']
    result1 = getPredictions(Teaching)
    course_content=request.GET['course_content']
    result2 = getPredictions(course_content)
    examination=request.GET['examination']
    result3 = getPredictions(examination)
    lab_work=request.GET['lab_work']
    result4 = getPredictions(lab_work)
    library=request.GET['library']
    result5 = getPredictions(library)
    extra=request.GET['extracurricular']
    result6 = getPredictions(extra)
    if result1=='Positive':
        r1=1
    elif result1=='Negative':
        r1=-1
    else:
        r1=0
    if result2=='Positive':
        r2=1
    elif result2=='Negative':
        r2=-1
    else:
        r2=0
    if result3=='Positive':
        r3=1
    elif result3=='Negative':
        r3=-1
    else:
        r3=0
    if result4=='Positive':
        r4=1
    elif result4=='Negative':
        r4=-1
    else:
        r4=0
    if result5=='Positive':
        r5=1
    elif result5=='Negative':
        r5=-1
    else:
        r5=0
    if result6=='Positive':
        r6=1
    elif result6=='Negative':
        r6=-1
    else:
        r6=0
    time = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
    with open('vv_web/datac.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([time,r1,Teaching,r2,course_content,r3,examination,r4,lab_work,r5,library,r6,extra])

    return(render(request,'thanks.html'))


def resultp(request):
    Teaching = request.GET['Teaching']
    r1 = getPredictionsP(Teaching)
    course_content = request.GET['course_content']
    r2 = getPredictionsP(course_content)
    examination = request.GET['examination']
    r3 = getPredictionsP(examination)
    lab_work = request.GET['lab_work']
    r4 = getPredictionsP(lab_work)
    library = request.GET['library']
    r5 = getPredictionsP(library)
    extra = request.GET['extracurricular']
    r6 = getPredictionsP(extra)
    time = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
    with open('vv_web/datap.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(
            [time, r1, Teaching, r2, course_content, r3, examination, r4, lab_work, r5, library, r6, extra])

    return render(request, 'thanks.html')

def resultj(request):
    Teaching = request.GET['Teaching']
    result1 = getPredictions(Teaching)
    course_content = request.GET['course_content']
    result2 = getPredictions(course_content)
    examination = request.GET['examination']
    result3 = getPredictions(examination)
    lab_work = request.GET['lab_work']
    result4 = getPredictions(lab_work)
    library = request.GET['library']
    result5 = getPredictions(library)
    extra = request.GET['extracurricular']
    result6 = getPredictions(extra)
    if result1 == 'Positive':
        r1 = 1
    elif result1 == 'Negative':
        r1 = -1
    else:
        r1 = 0
    if result2 == 'Positive':
        r2 = 1
    elif result2 == 'Negative':
        r2 = -1
    else:
        r2 = 0
    if result3 == 'Positive':
        r3 = 1
    elif result3 == 'Negative':
        r3 = -1
    else:
        r3 = 0
    if result4 == 'Positive':
        r4 = 1
    elif result4 == 'Negative':
        r4 = -1
    else:
        r4 = 0
    if result5 == 'Positive':
        r5 = 1
    elif result5 == 'Negative':
        r5 = -1
    else:
        r5 = 0
    if result6 == 'Positive':
        r6 = 1
    elif result6 == 'Negative':
        r6 = -1
    else:
        r6 = 0
    time = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
    with open('vv_web/dataj.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(
            [time, r1, Teaching, r2, course_content, r3, examination, r4, lab_work, r5, library, r6, extra])

    return (render(request, 'thanks.html'))

    # decide the file name
    # write the headers
def dd(request):
    wrapper = FileWrapper(open('vv_web/datac.csv', 'rb'))
    response = HttpResponse(wrapper, content_type='text/csv')
    response['Content-Disposition'] = 'inline; filename=' + os.path.basename('vv_web/datac.csv')
    return (response)

def ddp(request):
    wrapper = FileWrapper(open('vv_web/datap.csv', 'rb'))
    response = HttpResponse(wrapper, content_type='text/csv')
    response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path2)
    return response

def ddj(request):
    wrapper = FileWrapper(open('vv_web/dataj.csv', 'rb'))
    response = HttpResponse(wrapper, content_type='text/csv')
    response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path3)
    return response
def hom(request):
    return render(request,'hom.html')
def any(request):
    return render(request,'analysis.html')



def login(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    if username=='admin' and password=='admin':
        return render(request,'analysis.html')
    else:
        return render(request,'login.html')

def loginstd(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    if username=='student' and password=='12345':
        return render(request,'index.html')
    else:
        return render(request,'loginstd.html')


def contact(request):
    return render(request,'contact.html')


# our home page view

def getPredictionsP(pclass):
    model = pickle.load(open(r'C:\Users\jeeva\Downloads\SVM classifier.pkl', 'rb'))
    result= model.predict(pd.array([pclass]))
    return result[0]


def getPredictions(pclass):

    processed_data = prepare_data(pclass, tokenizer)
    result = make_prediction(sentiment_model, processed_data=processed_data)
    return result

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }


def make_prediction(model, processed_data, classes=['Neutral', 'Positive', 'Negative']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


def thanks(request):
    return render(request,'thanks.html')
def thnx(request):
    return render(request,'thnx.html')
def inx1(request):
    return render(request,'indexc.html')
def inx2(request):
    return render(request,'indexpython.html')
def inx3(request):
    return render(request,'indexjava.html')
def anyc(request):
    return render(request,'analysisc.html')
def anyp(request):
    return render(request,'analysispython.html')
def anyj(request):
    return render(request,'analysisjava.html')

def pie(request):
    path = 'vv_web/datac.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]
    dft=teaching_positive_count+teaching_negative_count+teaching_neutral_count
    tp = teaching_positive_count * 100
    tn = teaching_negative_count * 100
    tpn = teaching_neutral_count * 100

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]
    dfc=coursecontent_positive_count+coursecontent_negative_count+coursecontent_neutral_count
    cp = coursecontent_positive_count * 100
    cn = coursecontent_negative_count * 100
    cpn = coursecontent_neutral_count * 100

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]
    dfe = examination_positive_count+examination_negative_count+examination_neutral_count
    exp = examination_positive_count * 100
    exn = examination_negative_count * 100
    expn = examination_neutral_count * 100

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]
    dfl=labwork_positive_count+labwork_negative_count+labwork_neutral_count
    lbp = labwork_positive_count * 100
    lbn = labwork_negative_count * 100
    lbpn = labwork_neutral_count * 100

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]
    dfll=libraryfacilities_positive_count+libraryfacilities_negative_count+libraryfacilities_neutral_count
    lbrp = libraryfacilities_positive_count * 100
    lbrn = libraryfacilities_negative_count * 100
    lbrpn = libraryfacilities_neutral_count * 100

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]
    dfco=extracurricular_positive_count+extracurricular_negative_count+extracurricular_neutral_count
    cop = extracurricular_positive_count* 100
    con = extracurricular_negative_count * 100
    copn = extracurricular_neutral_count * 100

    total_positive_feedbacks = teaching_positive_count + coursecontent_positive_count + examination_positive_count + labwork_positive_count + libraryfacilities_positive_count + extracurricular_positive_count
    total_neutral_feedbacks = teaching_neutral_count + coursecontent_neutral_count + examination_neutral_count + labwork_neutral_count + libraryfacilities_neutral_count + extracurricular_neutral_count
    total_negative_feedbacks = teaching_negative_count + coursecontent_negative_count + examination_negative_count + labwork_negative_count + libraryfacilities_negative_count + extracurricular_negative_count
    ttlp = total_positive_feedbacks * 100
    ttln = total_negative_feedbacks * 100
    ttlpn = total_neutral_feedbacks * 100

    return render(request,'analysisc.html',{'dft':dft,'dfc':dfc,'dfe':dfe,'dfl':dfl,'dfll':dfll,'dfco':dfco,'ttlp':ttlp//total_feedbacks,'ttln':ttln//total_feedbacks,'ttlpn':ttlpn//total_feedbacks,'tp':tp//total_positive_feedbacks,'tn':tn//total_negative_feedbacks,'tpn':tpn//total_neutral_feedbacks,'cp':cp//total_positive_feedbacks,'cn':cn//total_negative_feedbacks,'cpn':cpn//total_neutral_feedbacks,'exp':exp//total_positive_feedbacks,'exn':exn//total_negative_feedbacks,'expn':expn//total_neutral_feedbacks,'lbp':lbp//total_positive_feedbacks,'lbn':lbn//total_negative_feedbacks,'lbpn':lbpn//total_neutral_feedbacks,'lbrp':lbrp//total_positive_feedbacks,'lbrn':lbrn//total_negative_feedbacks,'lbrpn':lbrpn//total_neutral_feedbacks,'cop':cop//total_positive_feedbacks,'con':con//total_negative_feedbacks,'copn':copn//total_neutral_feedbacks,'total_feedbacks':total_feedbacks,'total_positive_feedbacks':total_positive_feedbacks ,'teaching_positive_count':teaching_positive_count,'coursecontent_positive_count':coursecontent_positive_count, 'examination_positive_count':examination_positive_count,'labwork_positive_count':labwork_positive_count,'libraryfacilities_positive_count':libraryfacilities_positive_count,'extracurricular_positive_count':extracurricular_positive_count,'total_neutral_feedbacks':total_neutral_feedbacks,'teaching_neutral_count':teaching_neutral_count,'coursecontent_neutral_count':coursecontent_neutral_count,'examination_neutral_count':examination_neutral_count, 'labwork_neutral_count':labwork_neutral_count,'libraryfacilities_neutral_count':libraryfacilities_neutral_count, 'extracurricular_neutral_count':extracurricular_neutral_count,'total_negative_feedbacks':total_negative_feedbacks,'teaching_negative_count':teaching_negative_count,'coursecontent_negative_count':coursecontent_negative_count,'examination_negative_count':examination_negative_count,'labwork_negative_count':labwork_negative_count, 'libraryfacilities_negative_count':libraryfacilities_negative_count,'extracurricular_negative_count':extracurricular_negative_count})

def piep(request):
    path = 'vv_web/datap.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]
    dft = teaching_positive_count + teaching_negative_count + teaching_neutral_count
    tp = teaching_positive_count * 100
    tn = teaching_negative_count * 100
    tpn = teaching_neutral_count * 100

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]
    dfc = coursecontent_positive_count + coursecontent_negative_count + coursecontent_neutral_count
    cp = coursecontent_positive_count * 100
    cn = coursecontent_negative_count * 100
    cpn = coursecontent_neutral_count * 100

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]
    dfe = examination_positive_count + examination_negative_count + examination_neutral_count
    exp = examination_positive_count * 100
    exn = examination_negative_count * 100
    expn = examination_neutral_count * 100

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]
    dfl = labwork_positive_count + labwork_negative_count + labwork_neutral_count
    lbp = labwork_positive_count * 100
    lbn = labwork_negative_count * 100
    lbpn = labwork_neutral_count * 100

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]
    dfll = libraryfacilities_positive_count + libraryfacilities_negative_count + libraryfacilities_neutral_count
    lbrp = libraryfacilities_positive_count * 100
    lbrn = libraryfacilities_negative_count * 100
    lbrpn = libraryfacilities_neutral_count * 100

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]
    dfco = extracurricular_positive_count + extracurricular_negative_count + extracurricular_neutral_count
    cop = extracurricular_positive_count * 100
    con = extracurricular_negative_count * 100
    copn = extracurricular_neutral_count * 100

    total_positive_feedbacks = teaching_positive_count + coursecontent_positive_count + examination_positive_count + labwork_positive_count + libraryfacilities_positive_count + extracurricular_positive_count
    total_neutral_feedbacks = teaching_neutral_count + coursecontent_neutral_count + examination_neutral_count + labwork_neutral_count + libraryfacilities_neutral_count + extracurricular_neutral_count
    total_negative_feedbacks = teaching_negative_count + coursecontent_negative_count + examination_negative_count + labwork_negative_count + libraryfacilities_negative_count + extracurricular_negative_count
    ttlp = total_positive_feedbacks * 100
    ttln = total_negative_feedbacks * 100
    ttlpn = total_neutral_feedbacks * 100

    return render(request, 'analysispython.html',
                  {'dft': dft, 'dfc': dfc, 'dfe': dfe, 'dfl': dfl, 'dfll': dfll, 'dfco': dfco,
                   'ttlp': ttlp // total_feedbacks, 'ttln': ttln // total_feedbacks, 'ttlpn': ttlpn // total_feedbacks,
                   'tp': tp // total_positive_feedbacks, 'tn': tn // total_negative_feedbacks,
                   'tpn': tpn // total_neutral_feedbacks, 'cp': cp // total_positive_feedbacks,
                   'cn': cn // total_negative_feedbacks, 'cpn': cpn // total_neutral_feedbacks,
                   'exp': exp // total_positive_feedbacks, 'exn': exn // total_negative_feedbacks,
                   'expn': expn // total_neutral_feedbacks, 'lbp': lbp // total_positive_feedbacks,
                   'lbn': lbn // total_negative_feedbacks, 'lbpn': lbpn // total_neutral_feedbacks,
                   'lbrp': lbrp // total_positive_feedbacks, 'lbrn': lbrn // total_negative_feedbacks,
                   'lbrpn': lbrpn // total_neutral_feedbacks, 'cop': cop // total_positive_feedbacks,
                   'con': con // total_negative_feedbacks, 'copn': copn // total_neutral_feedbacks,
                   'total_feedbacks': total_feedbacks, 'total_positive_feedbacks': total_positive_feedbacks,
                   'teaching_positive_count': teaching_positive_count,
                   'coursecontent_positive_count': coursecontent_positive_count,
                   'examination_positive_count': examination_positive_count,
                   'labwork_positive_count': labwork_positive_count,
                   'libraryfacilities_positive_count': libraryfacilities_positive_count,
                   'extracurricular_positive_count': extracurricular_positive_count,
                   'total_neutral_feedbacks': total_neutral_feedbacks,
                   'teaching_neutral_count': teaching_neutral_count,
                   'coursecontent_neutral_count': coursecontent_neutral_count,
                   'examination_neutral_count': examination_neutral_count,
                   'labwork_neutral_count': labwork_neutral_count,
                   'libraryfacilities_neutral_count': libraryfacilities_neutral_count,
                   'extracurricular_neutral_count': extracurricular_neutral_count,
                   'total_negative_feedbacks': total_negative_feedbacks,
                   'teaching_negative_count': teaching_negative_count,
                   'coursecontent_negative_count': coursecontent_negative_count,
                   'examination_negative_count': examination_negative_count,
                   'labwork_negative_count': labwork_negative_count,
                   'libraryfacilities_negative_count': libraryfacilities_negative_count,
                   'extracurricular_negative_count': extracurricular_negative_count})


def piej(request):
    path = 'vv_web/dataj.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]
    dft = teaching_positive_count + teaching_negative_count + teaching_neutral_count
    tp = teaching_positive_count * 100
    tn = teaching_negative_count * 100
    tpn = teaching_neutral_count * 100

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]
    dfc = coursecontent_positive_count + coursecontent_negative_count + coursecontent_neutral_count
    cp = coursecontent_positive_count * 100
    cn = coursecontent_negative_count * 100
    cpn = coursecontent_neutral_count * 100

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]
    dfe = examination_positive_count + examination_negative_count + examination_neutral_count
    exp = examination_positive_count * 100
    exn = examination_negative_count * 100
    expn = examination_neutral_count * 100

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]
    dfl = labwork_positive_count + labwork_negative_count + labwork_neutral_count
    lbp = labwork_positive_count * 100
    lbn = labwork_negative_count * 100
    lbpn = labwork_neutral_count * 100

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]
    dfll = libraryfacilities_positive_count + libraryfacilities_negative_count + libraryfacilities_neutral_count
    lbrp = libraryfacilities_positive_count * 100
    lbrn = libraryfacilities_negative_count * 100
    lbrpn = libraryfacilities_neutral_count * 100

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]
    dfco = extracurricular_positive_count + extracurricular_negative_count + extracurricular_neutral_count
    cop = extracurricular_positive_count * 100
    con = extracurricular_negative_count * 100
    copn = extracurricular_neutral_count * 100

    total_positive_feedbacks = teaching_positive_count + coursecontent_positive_count + examination_positive_count + labwork_positive_count + libraryfacilities_positive_count + extracurricular_positive_count
    total_neutral_feedbacks = teaching_neutral_count + coursecontent_neutral_count + examination_neutral_count + labwork_neutral_count + libraryfacilities_neutral_count + extracurricular_neutral_count
    total_negative_feedbacks = teaching_negative_count + coursecontent_negative_count + examination_negative_count + labwork_negative_count + libraryfacilities_negative_count + extracurricular_negative_count
    ttlp = total_positive_feedbacks * 100
    ttln = total_negative_feedbacks * 100
    ttlpn = total_neutral_feedbacks * 100

    return render(request, 'analysisjava.html',
                  {'dft': dft, 'dfc': dfc, 'dfe': dfe, 'dfl': dfl, 'dfll': dfll, 'dfco': dfco,
                   'ttlp': ttlp // total_feedbacks, 'ttln': ttln // total_feedbacks, 'ttlpn': ttlpn // total_feedbacks,
                   'tp': tp // total_positive_feedbacks, 'tn': tn // total_negative_feedbacks,
                   'tpn': tpn // total_neutral_feedbacks, 'cp': cp // total_positive_feedbacks,
                   'cn': cn // total_negative_feedbacks, 'cpn': cpn // total_neutral_feedbacks,
                   'exp': exp // total_positive_feedbacks, 'exn': exn // total_negative_feedbacks,
                   'expn': expn // total_neutral_feedbacks, 'lbp': lbp // total_positive_feedbacks,
                   'lbn': lbn // total_negative_feedbacks, 'lbpn': lbpn // total_neutral_feedbacks,
                   'lbrp': lbrp // total_positive_feedbacks, 'lbrn': lbrn // total_negative_feedbacks,
                   'lbrpn': lbrpn // total_neutral_feedbacks, 'cop': cop // total_positive_feedbacks,
                   'con': con // total_negative_feedbacks, 'copn': copn // total_neutral_feedbacks,
                   'total_feedbacks': total_feedbacks, 'total_positive_feedbacks': total_positive_feedbacks,
                   'teaching_positive_count': teaching_positive_count,
                   'coursecontent_positive_count': coursecontent_positive_count,
                   'examination_positive_count': examination_positive_count,
                   'labwork_positive_count': labwork_positive_count,
                   'libraryfacilities_positive_count': libraryfacilities_positive_count,
                   'extracurricular_positive_count': extracurricular_positive_count,
                   'total_neutral_feedbacks': total_neutral_feedbacks,
                   'teaching_neutral_count': teaching_neutral_count,
                   'coursecontent_neutral_count': coursecontent_neutral_count,
                   'examination_neutral_count': examination_neutral_count,
                   'labwork_neutral_count': labwork_neutral_count,
                   'libraryfacilities_neutral_count': libraryfacilities_neutral_count,
                   'extracurricular_neutral_count': extracurricular_neutral_count,
                   'total_negative_feedbacks': total_negative_feedbacks,
                   'teaching_negative_count': teaching_negative_count,
                   'coursecontent_negative_count': coursecontent_negative_count,
                   'examination_negative_count': examination_negative_count,
                   'labwork_negative_count': labwork_negative_count,
                   'libraryfacilities_negative_count': libraryfacilities_negative_count,
                   'extracurricular_negative_count': extracurricular_negative_count})


def txtrpt(request):
    path = 'vv_web/datac.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]

    with open('report_file.txt', 'w') as f:
        f.write("C COURSE REPORT\n\n")
        f.write("Total Feedbacks:")
        total_feedbacks = str(total_feedbacks)
        f.write(total_feedbacks)
        f.write('\n')
        f.write("TEACHING:\n   ")
        f.write("Positive:")
        f.write(str(teaching_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(teaching_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(teaching_neutral_count))
        f.write('\n')
        f.write("COURSE CONTENT:\n   ")
        f.write("Positive:")
        f.write(str(coursecontent_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(coursecontent_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(coursecontent_neutral_count))
        f.write('\n')
        f.write("EXAMINATIONS:\n   ")
        f.write("Positive:")
        f.write(str(examination_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(examination_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(examination_neutral_count))
        f.write('\n')
        f.write("LAB_WORK:\n   ")
        f.write("Positive:")
        f.write(str(labwork_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(labwork_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(labwork_neutral_count))
        f.write('\n')
        f.write("LIBRARY:\n   ")
        f.write("Positive:")
        f.write(str(libraryfacilities_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(libraryfacilities_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(libraryfacilities_neutral_count))
        f.write('\n')
        f.write("CO-CURRICULAR:\n   ")
        f.write("Positive:")
        f.write(str(extracurricular_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(extracurricular_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(extracurricular_neutral_count))
        f.close()
        wrapper1 = FileWrapper(open('report_file.txt', 'rb'))
        response1 = HttpResponse(wrapper1, content_type='text/csv')
        response1['Content-Disposition'] = 'inline; filename=' + os.path.basename('report_file.txt')
    return response1


def txtrpt_p(request):
    path = 'vv_web/datap.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]

    with open('report_file.txt', 'w') as f:
        f.write("PYTHON COURSE REPORT\n\n")
        f.write("Total Feedbacks:")
        total_feedbacks = str(total_feedbacks)
        f.write(total_feedbacks)
        f.write('\n')
        f.write("TEACHING:\n   ")
        f.write("Positive:")
        f.write(str(teaching_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(teaching_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(teaching_neutral_count))
        f.write('\n')
        f.write("COURSE CONTENT:\n   ")
        f.write("Positive:")
        f.write(str(coursecontent_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(coursecontent_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(coursecontent_neutral_count))
        f.write('\n')
        f.write("EXAMINATIONS:\n   ")
        f.write("Positive:")
        f.write(str(examination_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(examination_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(examination_neutral_count))
        f.write('\n')
        f.write("LAB_WORK:\n   ")
        f.write("Positive:")
        f.write(str(labwork_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(labwork_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(labwork_neutral_count))
        f.write('\n')
        f.write("LIBRARY:\n   ")
        f.write("Positive:")
        f.write(str(libraryfacilities_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(libraryfacilities_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(libraryfacilities_neutral_count))
        f.write('\n')
        f.write("CO-CURRICULAR:\n   ")
        f.write("Positive:")
        f.write(str(extracurricular_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(extracurricular_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(extracurricular_neutral_count))
        f.close()
        wrapper1 = FileWrapper(open('report_file.txt', 'rb'))
        response1 = HttpResponse(wrapper1, content_type='text/csv')
        response1['Content-Disposition'] = 'inline; filename=' + os.path.basename('report_file.txt')
    return response1

def txtrpt_j(request):
    path = 'vv_web/dataj.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index) * 6

    df1 = df.groupby('teachingscore').count()[['teaching']]
    teaching_negative_count = df1['teaching'][-1]
    teaching_neutral_count = df1['teaching'][0]
    teaching_positive_count = df1['teaching'][1]

    df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = df1['coursecontent'][-1]
    coursecontent_neutral_count = df1['coursecontent'][0]
    coursecontent_positive_count = df1['coursecontent'][1]

    df1 = df.groupby('examinationscore').count()[['examination']]
    examination_negative_count = df1['examination'][-1]
    examination_neutral_count = df1['examination'][0]
    examination_positive_count = df1['examination'][1]

    df1 = df.groupby('labworkscore').count()[['labwork']]
    labwork_negative_count = df1['labwork'][-1]
    labwork_neutral_count = df1['labwork'][0]
    labwork_positive_count = df1['labwork'][1]

    df1 = df.groupby('libraryfacilitiesscore').count()[['libraryfacilities']]
    libraryfacilities_negative_count = df1['libraryfacilities'][-1]
    libraryfacilities_neutral_count = df1['libraryfacilities'][0]
    libraryfacilities_positive_count = df1['libraryfacilities'][1]

    df1 = df.groupby('extracurricularscore').count()[['extracurricular']]
    extracurricular_negative_count = df1['extracurricular'][-1]
    extracurricular_neutral_count = df1['extracurricular'][0]
    extracurricular_positive_count = df1['extracurricular'][1]

    with open('report_file.txt', 'w') as f:
        f.write("JAVA COURSE REPORT\n\n")
        f.write("Total Feedbacks:")
        total_feedbacks = str(total_feedbacks)
        f.write(total_feedbacks)
        f.write('\n')
        f.write("TEACHING:\n   ")
        f.write("Positive:")
        f.write(str(teaching_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(teaching_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(teaching_neutral_count))
        f.write('\n')
        f.write("COURSE CONTENT:\n   ")
        f.write("Positive:")
        f.write(str(coursecontent_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(coursecontent_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(coursecontent_neutral_count))
        f.write('\n')
        f.write("EXAMINATIONS:\n   ")
        f.write("Positive:")
        f.write(str(examination_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(examination_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(examination_neutral_count))
        f.write('\n')
        f.write("LAB_WORK:\n   ")
        f.write("Positive:")
        f.write(str(labwork_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(labwork_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(labwork_neutral_count))
        f.write('\n')
        f.write("LIBRARY:\n   ")
        f.write("Positive:")
        f.write(str(libraryfacilities_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(libraryfacilities_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(libraryfacilities_neutral_count))
        f.write('\n')
        f.write("CO-CURRICULAR:\n   ")
        f.write("Positive:")
        f.write(str(extracurricular_positive_count))
        f.write('\n   ')
        f.write("Negative:")
        f.write(str(extracurricular_negative_count))
        f.write('\n   ')
        f.write("Neutral:")
        f.write(str(extracurricular_neutral_count))
        f.close()
        wrapper1 = FileWrapper(open('report_file.txt', 'rb'))
        response1 = HttpResponse(wrapper1, content_type='text/csv')
        response1['Content-Disposition'] = 'inline; filename=' + os.path.basename('report_file.txt')
    return response1
