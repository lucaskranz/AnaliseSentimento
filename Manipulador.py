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


#Remove os radicais repetidos e coloca as palavras em uma lista ordenada
#  pela frequência que as mesmas aparecem nas frases.
def busca_palavras_frases(frases):
    todas_palavras = []
    for (palavras,sentimento) in frases:
        todas_palavras.extend(palavras)
    return todas_palavras



#Extrai a distribuição de frequência de cada radical dentro da lista.
def busca_frequencia_palavras(nltk, lista_palavras):
    lista_palavras = nltk.FreqDist(lista_palavras)
    return lista_palavras


#Palavras únicas.
def busca_palavras_unicas(lista_frequencia):
    frequencia = lista_frequencia.keys()
    return frequencia


#Lê o arquivo e remove a  a primeira linha se a mesma for um cabeçalho.
def busca_base(nome_arquivo):
    arquivo = csv.reader(open(nome_arquivo))
    base = []
    for row in arquivo:
        if (row[5] != "texto"):
            base.append((row[5], row[6]))
    return base

