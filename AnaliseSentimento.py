# coding=utf-8
import nltk
import Manipulador

base_arq = Manipulador.busca_base("teste.csv")

stemmer = nltk.stem.RSLPStemmer()
frases = []
stop_words = Manipulador.get_stopwords()

print ("processando arquivo...")

for (palavras, sentimento) in base_arq:
    #print palavras
    filtrado = [str(stemmer.stem(Manipulador.remover_acentos(e))) for e in palavras.split() if e not in stop_words]
    #print filtrado
    frases.append((filtrado, sentimento))

palavras = Manipulador.busca_palavras_frases(frases)

frequencia = Manipulador.busca_frequencia_palavras(nltk, palavras)

palavras_caracteristicas = Manipulador.busca_palavras_unicas(frequencia)

print ("extraindo características...")
def extrator_caracteristicas(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavras_caracteristicas:
        caracteristicas['contem(%s)' % palavra] = (palavra in doc)
    return caracteristicas

#caracteristicas_frase = extrator_caracteristicas(['tim','gole','nov'])
#print(caracteristicas_frase)

print ("treinando algorítimo...")
treino = nltk.apply_features(extrator_caracteristicas, frases)

classificador = nltk.NaiveBayesClassifier.train(treino)

#frases para teste.

	#negativo
#teste = '@submarino Vocês protegem um fraudador!!! #Vergonha http://t.co/dR3hgnFh'
#teste = 'a rede globo é uma bosta'
	#positivo
teste = 'Olha! Tenho o videobit 62 do videoalbum Claudia Leitte #IssoVaiColar'
#teste = 'o programa panico me segue'


print ("Classificando...")

teste_steemming = []
for (palavras) in teste.split():
    filtrado = [e for e in palavras.split()]
    teste_steemming.append(str(stemmer.stem(Manipulador.remover_acentos(filtrado[0]))))


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