import nltk
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import Vocabulary, MLE, Lidstone
##nltk.download('punkt')

texto = """ainda que mal pergunte
ainda que mal respondas
ainda que mal te entenda
ainda que mal repitas  
"""

texto = texto.lower().split('\n')

## TOKENIZANDO O CORPUS
texto_tok = []
for verso in texto:
    tokens = nltk.word_tokenize(verso, language="portuguese")
    texto_tok.append(tokens)

##print(texto_tok)

##INSERINDO MARCADORES DE INICIO E FIM DE SENTENCA
texto_padded = []
ngramas = 2
for verso in texto_tok:
    padded = pad_both_ends(verso, 2)
    texto_padded.append(list(padded))
    
##print(texto_padded)

##CALCULANDO BIGRAMAS
ngramas = 2

bigramas_pad = []
for verso in texto_padded:
    bigramas = nltk.ngrams(verso, ngramas)
    bigramas_pad.append(list(bigramas))
    
bigramas_pad = list(flatten(bigramas_pad))
##print(bigramas_pad)

tokens = list(flatten(texto_padded))
##print(tokens)

##CRIANDO VOCABULARIO
vocab = Vocabulary(tokens, unk_cutoff=1)

##SIMPLIFICANOD O PRE-PROCESSAMENTO E TREINANDO UM MODELO DE LINGUAGEM
k=0.1
ngramas_pad, vocab = padded_everygram_pipeline(ngramas, texto_tok)
lm = Lidstone(order=ngramas, gamma=k)
lm.fit(ngramas_pad, vocab)

##GERANDO FRASES
##print(lm.generate(4, text_seed=["repitas"]))

##CALCULANDO A PROBABILIDADE LOGARITIMICA
value = lm.score("ainda que mal insista")
print(value)