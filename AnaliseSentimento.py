# coding=utf-8
import nltk
import Manipulador

stemmer = nltk.stem.RSLPStemmer()
stop_words = Manipulador.get_stopwords()
base_treino = Manipulador.busca_base("teste.csv")
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
verdadeiroPositivos = 0.0
verdadeiroNegativos = 0.0
falsoPositivos = 0.0
falsoNegativos = 0.0
f1 = 0.0

for (texto,sentimento) in base_teste:
    for palavras in texto.split():
        filtrado = [e for e in palavras.split()]
        teste_steemming.append(str(stemmer.stem(Manipulador.remover_acentos(filtrado[0]))))
    sent = classificador.classify(extrator_caracteristicas(teste_steemming))
    if (sent == '1') and (sentimento == sent):
        verdadeiroPositivos += 1.0
    elif (sent == '-1') and (sentimento == sent):
        verdadeiroNegativos += 1.0
    elif (sent == '1') and (sentimento != sent):
        falsoPositivos += 1.0
    elif (sent == '-1') and (sentimento != sent):
        falsoNegativos += 1.0


# print ("verdadeiroPositivos: " + str(verdadeiroPositivos))
# print ("falsoNegativos: " + str(falsoNegativos))
# print ("verdadeiroNegativos: " + str(verdadeiroNegativos))
# print ("falsoPositivos: " + str(falsoPositivos))
precisao = 0.0
revocacao = 0.0


#Precisão = (VP)/(VP+FP)
if verdadeiroPositivos + falsoPositivos > 0:
    precisao = (verdadeiroPositivos) / (verdadeiroPositivos + falsoPositivos)
    print ("Precisão: " + str(precisao * 100))

#Revocação = (VP)/(VP+FN)
if verdadeiroPositivos + falsoNegativos > 0:
    revocacao = (verdadeiroPositivos) / (verdadeiroPositivos + falsoNegativos)
    print ("Revocação: " + str(revocacao * 100))

if precisao + revocacao > 0:
    f1 = (((2 * precisao * revocacao) / (precisao + revocacao)) * 100)

#Acurácia = (VP+VN)/total
acuracia = (verdadeiroPositivos + verdadeiroNegativos) / len(base_teste)
print ("Acurácia: " + str(acuracia * 100))

print ("F1: " + str(f1))

