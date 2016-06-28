# coding=utf-8
import nltk
import Manipulador

stemmer = nltk.stem.RSLPStemmer()
stop_words = Manipulador.get_stopwords()
base_treino = Manipulador.busca_base("sementes_treinamento.csv")
base_teste = Manipulador.busca_base("sementes_treinamento.csv")

print ("processando arquivo...")

def filtrar_frases(base):
    frases_base = []
    for (palavras, sentimento) in base:
        filtrado = [str(stemmer.stem(Manipulador.remover_acentos(e))) for e in palavras.split() if e not in stop_words]
        frases_base.append((filtrado, sentimento))
    return frases_base

frases_treino = filtrar_frases(base_treino)

palavras = Manipulador.busca_palavras_frases(frases_treino)

frequencia = Manipulador.busca_frequencia_palavras(nltk, palavras)

palavras_caracteristicas = Manipulador.busca_palavras_unicas(frequencia)

print ("extraindo características...")
def extrator_caracteristicas(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavras_caracteristicas:
        caracteristicas['contem(%s)' % palavra] = (palavra in doc)
    return caracteristicas

print ("treinando algoritmo...")
treino = nltk.apply_features(extrator_caracteristicas, frases_treino)

classificador = nltk.NaiveBayesClassifier.train(treino)
print ("Classificando...\n")

teste_steemming = []
base_resultado = []
verdadeiroPositivos = 0.0
verdadeiroNegativos = 0.0
falsoPositivos = 0.0
falsoNegativos = 0.0
f1 = 0.0

for (texto,sentimento) in base_teste:
    for palavras in texto.split():
        filtrado = [e for e in palavras.split()]
        #teste_steemming.append(str(stemmer.stem(Manipulador.remover_acentos(filtrado[0]))))
        teste_steemming.append(str(stemmer.stem(Manipulador.remover_acentos(e))) for e in palavras.split() if e not in stop_words)
    sent = classificador.classify(extrator_caracteristicas(teste_steemming))
    if sent == '1' and sent == sentimento:
        verdadeiroPositivos += 1.0
    if sent == '-1' and sent == sentimento:
        verdadeiroNegativos += 1.0
    if sent == '1' and sent != sentimento:
        falsoPositivos += 1.0
    if sent == '-1' and sent != sentimento:
        falsoNegativos += 1.0


    # if sent == '1':
    #     if
    # sent == sentimento:
    # verdadeiroPositivos += 1.0
    # else:
    # falsoNegativos += 1.0
    # else:
    # if sent == sentimento:
    #     verdadeiroNegativos += 1.0
    # else:
    #     falsoPositivos += 1.0

print ("verdadeiroPositivos: " + str(verdadeiroPositivos))
print ("falsoNegativos: " + str(falsoNegativos))
print ("verdadeiroNegativos: " + str(verdadeiroNegativos))
print ("falsoPositivos: " + str(falsoPositivos))

precisao = (verdadeiroPositivos)/(verdadeiroPositivos+falsoPositivos)
revocacao = (verdadeiroPositivos)/(verdadeiroPositivos+falsoNegativos)
f1 = (((2 * precisao * revocacao) / (precisao + revocacao)) * 100)

#Precisão = (VP)/(VP+FP)
print ("Precisão: " + str(precisao))

#Revocação = (VP)/(VP+FN)
print ("Revocação: " + str(revocacao))

#F1
print ("F1: " + str(f1))
