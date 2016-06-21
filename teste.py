import nltk

base = [('tive uma ideia genial', 'alegria'),
        ('agora vou comer uma feijoada gostosa', 'alegria'),
        ('ganhei um baita desconto no carro', 'alegria'),
        ('estou ganhando a aposta', 'alegria'),
        ('meu time goleou novamente', 'alegria'),
        ('consegui arrumar todo aquele estrago', 'alegria'),
        ('nossa nova ideia deu errado', 'tristeza'),
        ('estou sem dinheiro para comer uma feijoada', 'tristeza'),
        ('perdi novamente a aposta', 'tristeza'),
        ('a fila no banco esta muito grande', 'tristeza')]
stopWords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles',
             'de', 'deu', 'do', 'e', 'estou', 'esta', 'ir', 'meu',
             'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para',
             'que', 'sem', 'talvez', 'tem', 'tendo', 'tenha', 'tive,'
                                                              'todo', 'um', 'uma', 'umas', 'uns', 'vou', 'vos']

stemmer = nltk.stem.RSLPStemmer()
frases = []
for (palavras, sentimento) in base:
    filtrado = [str(stemmer.stem(e)) for e in palavras.split() if e not in stopWords]
    frases.append((filtrado, sentimento))

def busca_palavras_frases(frases):
    todas_palavras = []
    for (palavras,sentimento) in frases:
        todas_palavras.extend(palavras)
    return todas_palavras
palavras = busca_palavras_frases(frases)

def busca_frequencia_palavras(lista_palavras):
    lista_palavras = nltk.FreqDist(lista_palavras)
    return lista_palavras
frequencia = busca_frequencia_palavras(palavras)

def busca_palavras_unicas(lista_frequencia):
    frequencia = lista_frequencia.keys()
    return frequencia
palavras_caracteristicas = busca_palavras_unicas(frequencia)

def extrator_caracteristicas(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavras_caracteristicas:
        caracteristicas['contem(%s)' % palavra] = (palavra in doc)
    return caracteristicas
caracteristicas_frase = extrator_caracteristicas(['tim','gole','nov'])

treino = nltk.apply_features(extrator_caracteristicas, frases)
classificador = nltk.NaiveBayesClassifier.train(treino)

teste = 'ganhei dinheiro na aposta'
teste_steemming = []
for (palavras) in teste.split():
    filtrado = [e for e in palavras.split()]
    teste_steemming.append(str(stemmer.stem(filtrado[0])))
print classificador.classify(extrator_caracteristicas(teste_steemming))

