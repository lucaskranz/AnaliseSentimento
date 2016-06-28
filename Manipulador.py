# coding=utf-8
import csv
from unicodedata import normalize
from nltk.corpus import stopwords

caracteres_invalidos = ['·', '–', '""', '¬', '#', '“', '"', '.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']


def remover_caracteres_especiais(txt):
    for caracter in caracteres_invalidos:
        txt = txt.replace(caracter, ' ')
    return txt


def remover_acentos(txt, codif='utf-8'):
    txt = remover_caracteres_especiais(txt)
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore')


# concatena caracteres especiais com as stop words
def get_stopwords(idioma = 'portuguese'):
    stop_words_portuges = stopwords.words(idioma)
    return stop_words_portuges


def busca_palavras_frases(frases):
    todas_palavras = []
    for (palavras,sentimento) in frases:
        todas_palavras.extend(palavras)
    return todas_palavras


def busca_frequencia_palavras(nltk, lista_palavras):
    lista_palavras = nltk.FreqDist(lista_palavras)
    return lista_palavras


def busca_palavras_unicas(lista_frequencia):
    frequencia = lista_frequencia.keys()
    return frequencia


def busca_base(nome_arquivo):
    arquivo = csv.reader(open(nome_arquivo))
    base = []
    a = 1
    for row in arquivo:
        if (row[5] != "texto"):
            base.append((row[5], row[6]))
        a=a+1
    return base





