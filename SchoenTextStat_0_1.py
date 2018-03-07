# coding: utf-8

import pymorphy2
import re
import string
import os
import time
import collections as cl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages


def numnum(y):
    return sum([i[1] for i in y])

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
def hihi(l, cc, s, mc=None, kost=None):
    labels, values = zip(*l)
    indexes = np.arange(len(labels))
    plt.plot(indexes, values, cc)
    plt.grid(axis = 'x', color = 'k', linestyle=':')
    plt.grid(axis = 'y', color ='k', linestyle=':')
    plt.title(s, fontsize = 16, loc = 'left', fontweight='light')
    if not mc:
        plt.ylabel('количество употреблений')
        plt.xlabel('количество слов')
    else:
        plt.ylabel('количество употреблений')
        plt.xticks(indexes, labels, fontsize=10)
        
def pipi(w, k, s):
    y = cl.Counter(w[k]).most_common()
    z1, v1 = zip(*y)
    plt.pie(v1, labels=z1, autopct='%1.1f%%', startangle=90, 
            colors=['#df2020', 'orange', 'w'])
    plt.title(s, fontsize = 16, loc = 'left', fontweight='light')
    
def fifi(ww, s, cc, k=None):
    if k:
        NounWerte = cl.Counter(ww[k]).most_common()
        NN = numnum(NounWerte)
    else:
        l = []
        for ke in ww:
            n = [ke for i in range(len(ww[ke]))]
            l.extend(n)
        NounWerte = cl.Counter(l).most_common()
        NN = numnum(NounWerte)
    Etiketten, Werte = zip(*NounWerte)
    pos = np.arange(len(Werte))
    plt.barh(pos, Werte, color = cc)
    for p, W in zip(pos, Werte):
        plt.annotate(str(W)+' – '+str(nn(NN, W)), xy=(W + 1, p), fontsize = 10, 
                     va='center')
    plt.yticks(pos, Etiketten, fontsize = 10)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))
    remove_border(left=False, bottom=False)
    plt.grid(axis = 'x', color ='w', linestyle=':')
    plt.gca().invert_yaxis()
    plt.title(s, fontsize = 16, loc = 'left', fontweight='light')

def tete(s=None, n=None, ss=None):
    if type(n) == int:
        if n < 99999:
            plt.text(0,.4,str(n), fontsize=85, fontweight='light')
        else:
            plt.text(0,.4,'>'+str(int(n/1000))+'k', fontsize=85, 
            fontweight='light')
            if ss:
                ss2 = ss
                ss = str(n)
                ss = ss + '\n' + '**' + ss2
            else:
                ss = str(n) + '\n'
    else:
        plt.text(0,.4,str(n), fontsize=85, fontweight='light')
    plt.text(0,.7,s, fontsize=18, fontweight='light', )
    if ss:
        plt.text(0,.2,'*'+ss, 
             fontsize=18, fontweight='light', color='#df2020')
    plt.xticks([])
    plt.yticks([])
    remove_border(left=False, bottom=False)

Interpunktion = (string.punctuation+'\u2014\u2013\u2012\u2010\u2212'+
                 '«»‹›‘’“”„`…–')
Wort_Tokenize = re.compile(r"([^\w_\u2019\u2010\u002F-]|[+])")
    
def Word_Tokenizer(text):
    return [t.lower() for t in Wort_Tokenize.split(text) 
            if t and not t.isspace() and not t in Interpunktion 
            and not any(map(str.isdigit, t))]

def Satz(n):
    S = set()
    [S.update(set(i)) for i in n.values()] 
    return S

nn = lambda x, y: '%.1f%%' %((100/x)*y)


def main():
    while True:
        ms = {'s':'Small text', 'b':'Big text'}
        print('Choose mode.\n\ns - for ' + ms['s'] + '\n\nb - for ' + ms['b'])
        Mode = input()
        if Mode in ms:
            print('\n\n***SchoenTextStat is in ' + ms[Mode] + ' mode***\n\n')
            break
        else:
            print('\nThere is no such mode')

    Pfad = input('Path to the file: ')
    RName = os.path.basename(Pfad)
    
    if not os.path.isfile(Pfad):
        print("\n~~~~~~~~~~~~~~~~~~~~~~File doesn't exist.~~~~~~~~~~~~~~~~~~~~~~\n\n\n", 
            "I see my path, but I don't know where it leads.\n",
            "Not knowing where I'm going is what inspires me to travel it.",
            "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        pass

    Starten = time.time()

    with open(Pfad, 'r', encoding='utf-8') as f:
        Gross_Liste = Word_Tokenizer(f.read())

    Morph = pymorphy2.MorphAnalyzer()

    Wordforms = set(Gross_Liste)
    Gefaelschtes = []
    OMO = []
    Inde = []

    Lemmas_Woerterbuch = defaultdict(set)
    Wordforms_Woerterbuch = defaultdict(list)
    Faelle_Woerterbuch = defaultdict(list)
    Verben_Woerterbuch = defaultdict(list)

    for Wort in Wordforms:
        if 'ё' in Wort:
            Wort = Wort.replace('ё','е')
        MP_W = Morph.parse(Wort)
        Informationen = MP_W[0]
        IT = Informationen.tag
        Wortart = IT.POS
        if Wortart:
            if len(MP_W) > 1:
                o = [Wort, len(MP_W)]
                OMO.append(o)
            Inde.append(Informationen.score)

            Lemma = Informationen.normal_form
            if str(Informationen.methods_stack[0][0]) == '<FakeDictionary>':
                m = [Wort, IT.cyr_repr, Informationen.normal_form, round(Informationen.score, 2)]
                Gefaelschtes.append(m)
            Wordforms_Woerterbuch[Morph.lat2cyr(Wortart)].append(Wort)
            Lemmas_Woerterbuch[Morph.lat2cyr(Wortart)].add(Lemma) 
            if Wortart == 'NOUN' or Wortart == 'ADJF':
                Case = IT.case
                Faelle_Woerterbuch[Morph.lat2cyr(Wortart)].append(Morph.lat2cyr(Case))
            if Wortart == 'VERB':
                Nummer, Zeit, Person = IT.number, IT.tense, IT.person
                if Nummer:
                    Verben_Woerterbuch['Число'].append(Morph.lat2cyr(Nummer))
                if Zeit:
                    Verben_Woerterbuch['Время'].append(Morph.lat2cyr(Zeit))
                if Person:
                    Verben_Woerterbuch['Лицо'].append(Morph.lat2cyr(Person))

    Gross_Nummer = len(Gross_Liste) #Словоупотреблений
    Wordforms_Nummer = len(Wordforms) #Словоформ (с проблемными) 
    Lemmas_Nummer = len(Satz(Lemmas_Woerterbuch)) #Лемм по множеству
    Prozentsatz_dem_Lemmas = nn(Wordforms_Nummer, Lemmas_Nummer)
    Index_des_Reichtums = round((Lemmas_Nummer/Wordforms_Nummer), 2)
    
    with open('stop_words.txt', 'r', encoding='utf-8') as f:
        Stop = f.read().split()
    
    Nein_Stop = []

    for i in cl.Counter(Gross_Liste).most_common():
        if not i[0] in Stop:
            Nein_Stop.append(i)

    GL_C = cl.Counter(Gross_Liste).most_common()

    if len(GL_C) > 10:
        gl = []
        for i in GL_C:
            gl.append(i)
            if i[1] == 1:
                GL_C = gl
                break
    
    if not os.path.isdir('./rusult'):
        os.mkdir('./rusult')

    pp = PdfPages('./rusult/rusult_'+RName+'.pdf')

    fig = plt.figure(figsize=(16,6))
    tete('', 'Результат\nанализа текста', RName)
    pp.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(16,4.65))
    plt.subplot(1,3,1)
    tete('Количество\nсловоупотреблений', Gross_Nummer)
    plt.subplot(1,3,2)
    tete('Количество\nсловоформ', Wordforms_Nummer)
    plt.subplot(1,3,3)
    tete('Количество\nлемм', Lemmas_Nummer, Prozentsatz_dem_Lemmas+
         ' от числа\nсловоформ')
    pp.savefig(fig)
    plt.close()

    if Mode == 'b':
        fig = plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        hihi(GL_C, '#df2020', 'Характер распределения\nслов в тексте (max, min)', kost=True)
        plt.subplot(1,2,2)
        hihi(GL_C[:10], '#df2020', 'Распределение '+str(len(GL_C[:10]))+
            ' наиболее частых\nслов в тексте', 
             mc=True, kost=True)
        top10 = GL_C[:10]
        yy = .75*top10[0][1]
        top10N = sum([i[1] for i in top10])
        plt.text(4, yy, '*'+str(nn(Gross_Nummer, top10N))+
                 '\nот общего\nколичества слов', 
                 fontsize=25, color='black', fontweight='light')
        pp.savefig(fig)
        plt.close()
    if Mode == 's':
        fig = plt.figure(figsize=(16,6))
        hihi(GL_C, '#df2020', 'Характер распределения\nслов в тексте (max, min)', kost=True)
        pp.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=(16,6))
        hihi(GL_C[:10], '#df2020', 'Распределение '+str(len(GL_C[:10]))+
            ' наиболее частых\nслов в тексте', 
             mc=True, kost=True)
        top10 = GL_C[:10]
        yy = .75*top10[0][1]
        top10N = sum([i[1] for i in top10])
        plt.text(4, yy, '*'+str(nn(Gross_Nummer, top10N))+
                 '\nот общего\nколичества слов', 
                 fontsize=25, color='black', fontweight='light')
        pp.savefig(fig)
        plt.close()
    
    LNS = len(Nein_Stop)
    targ = 10
    if LNS > 0:
        if LNS < 10:
            targ = LNS

        fig = plt.figure(figsize=(16,6))
        hihi(Nein_Stop[:targ], '#df2020', 
             'Распределение '+str(targ)+
             ' наиболее частых\nслов в тексте (без стоп-слов)', 
             mc=True, kost=True)
        yy2 = .75*Nein_Stop[0][1]
        Nein_GN = sum([i[1] for i in Nein_Stop])
        Nein_StopN = sum([i[1] for i in Nein_Stop[:targ]])
        plt.text(7, yy2, '*'+str(nn(Gross_Nummer, Nein_GN))+
                 '\nот общего\nколичества', 
                 fontsize=25, color='black', fontweight='light')
        pp.savefig(fig)
        plt.close()

    fig = plt.figure(figsize=(16,6))
    fifi(Wordforms_Woerterbuch, 'Распределение cловоформ\nпо частям речи', 
         '#df2020')
    pp.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(16,6))
    fifi(Lemmas_Woerterbuch, 
        'Распределение лемм\nпо частям речи', 'orange')
    pp.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(16,4.65))
    plt.subplot(1,2,1)
    tete(u"Коэффициент\nлексического богатства текста", Index_des_Reichtums, 
         'отношение числа различных\nлемм к общему числу словоформ')
    plt.subplot(1,2,2)
    tete(u"Процент\nомонимичных словоформ", nn(Wordforms_Nummer, len(OMO)))
    pp.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(16,6))
    Ue = []
    for k, v in Faelle_Woerterbuch.items():
        for kk, vv in cl.Counter(v).items():
            lll = [vv, kk, k]
            Ue.append(lll)

    NNz1 = max([numnum(cl.Counter(Faelle_Woerterbuch['СУЩ']).most_common()), 
                numnum(cl.Counter(Faelle_Woerterbuch['ПРИЛ']).most_common())])        
    NNz2 = min([numnum(cl.Counter(Faelle_Woerterbuch['СУЩ']).most_common()), 
                numnum(cl.Counter(Faelle_Woerterbuch['ПРИЛ']).most_common())]) 

    Ue.sort(reverse=True)
    df = pd.DataFrame(Ue, columns=['кол', 'п', 'чр'])
    g = sns.barplot(y='п', x='кол', hue='чр', data=df, 
                    palette=['red','black'])

    plt.grid(axis='x', color='white', linestyle=':')
    remove_border(left=False, bottom=False)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))

    for p in ax.patches:
        if p.get_width()==p.get_width():
            nnn = round(p.get_width())
            if p.get_facecolor() == (0.875, 0.125, 0.125, 1.0):
                boba = str(nnn)+' – '+str(nn(NNz1, nnn))
            else:
                boba = str(nnn)+' – '+str(nn(NNz2, nnn))
            ax.annotate(boba, (p.get_width(), p.get_y()+p.get_height()/2), 
                        va='center', fontsize=10, xytext=(2, 0), 
                        textcoords='offset points')

    ax.legend(ncol=2, loc="lower center")
    ax.set(ylabel="", xlabel="")
    plt.title('Распределение падежей\nсуществительных и прилагательных', 
              fontsize = 16, loc = 'left', fontweight='light')
    pp.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(16,4.65))
    plt.subplot(1,3,1)
    pipi(Verben_Woerterbuch, 'Число', 'Распределение категории числа\nу глаголов')
    plt.subplot(1,3,2)
    pipi(Verben_Woerterbuch, 'Лицо', 'Распределение категории лица\nу глаголов')
    plt.subplot(1,3,3)
    pipi(Verben_Woerterbuch, 'Время', 'Распределение категории времени\nу глаголов')
    pp.savefig(fig)
    
    if Gefaelschtes:
        df1 = pd.DataFrame(Gefaelschtes, columns=['слово', 'разбор','лема', 'оценка точности разбора'])
        df1.to_csv('./rusult/unknown_words_'+RName+'.csv', encoding='utf-8')

        gfg = [i[0] for i in Gefaelschtes]
        h = ''
        u = 0
        RaRa = len(gfg)//4+1
        for i in range(18):
            m = ''
            for e in gfg[u:u+3]:
                m += e + ' | '
            h += m + '\n'
            u+=3
        if RaRa > 18:
            h += '<...>'
        fig = plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        tete(u"Процент слов,\nотсутствующих в словаре", nn(Wordforms_Nummer, len(gfg)), 
            'таблица несловарных слов\nсодержится в файле\n./rusult/unknown_words_')
        plt.subplot(1,2,2)        
        plt.text(-.2, 0,  h, fontsize=14, fontweight='light')
        plt.xticks([])
        plt.yticks([])
        remove_border(left=False, bottom=False)
        pp.savefig(fig)
        plt.close()
    else:
        fig = plt.figure(figsize=(16,4,65))
        tete(u"Несловарные слова в тексте отсутствуют")
        pp.savefig(fig)
        plt.close()

    fig = plt.figure(figsize=(16,4.65))
    tete(u"Средняя оценка точности разбора\nкаждого сова", str(round((sum(Inde)/len(Inde)), 2)), 
        'по данным pymorphy2')
    pp.savefig(fig)
    plt.close()

    pp.close()

    print('\nAnalysis is done!\n\nResult in ./rusult/ directory.\n\nBitte schoen.')
    print("\ntime: " + str(round((time.time()-Starten), 2)))

main()
