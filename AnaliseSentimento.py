# coding=utf-8
import nltk
import Manipulador

stemmer = nltk.stem.RSLPStemmer()
stop_words = Manipulador.get_stopwords()
base_treino = Manipulador.busca_base("treino.csv")
base_teste = Manipulador.busca_base("teste.csv")


print ("processando arquivo...")


def filtrar_frases(base):
    frases_base = []
    for (palavras, sentimento) in base:
        filtrado = [str(stemmer.stem(Manipulador.remover_acentos(e))) for e in palavras.split() if e not in stop_words]
        frases_base.append((filtrado, sentimento))
    return frases_base


#frases = []
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

print ("treinando algorítimo...")
treino = nltk.apply_features(extrator_caracteristicas, frases_treino)

classificador = nltk.NaiveBayesClassifier.train(treino)

print ("Classificando...")

teste = filtrar_frases(base_teste)
teste_steemming = []

#arrumar aqui nos laços.
for linha in teste:
    for palavras in linha:
        teste_steemming.append((stemmer.stem(Manipulador.remover_acentos(palavras[0])), '0'))

for linha_classificador in teste_steemming:
    teste_steemming[1] = classificador.classify(extrator_caracteristicas(linha_classificador[0]))

for linha_classificador in teste_steemming:
    print classificador.classify(extrator_caracteristicas(linha_classificador[0])) + "valor: " + linha_classificador[0]

if classificador.classify(extrator_caracteristicas(teste_steemming)) == '1':
    print ("Twit Positivo")
else:
    print ("Twit Negativo")

#--------------------------------------------
#buscar a precisão/revocação/F1
#nltk.precision()


#--------------------------------------------

#print classificador.classify(extrator_caracteristicas(teste_steemming))
#print classificador.show_most_informative_features(5)


#teste = 'perdi dinheiro na aposta'
#teste = 'estou com muita dor nos meu dedos de frio'

#frases para teste.

#negativo
#teste = '@submarino Vocês protegem um fraudador!!! #Vergonha http://t.co/dR3hgnFh'
#teste = 'a rede globo é uma bosta'
#positivo
#teste = 'Olha! Tenho o videobit 62 do videoalbum Claudia Leitte #IssoVaiColar'
#teste = 'o programa panico me segue'
