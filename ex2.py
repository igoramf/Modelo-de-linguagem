import nltk
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import Vocabulary, MLE, Lidstone
##nltk.download('punkt')

texto = """Eu não sei dizer se este amor
Vai voltar a ser o que já foi
Não sei se depois que amanhecer
Nós vamos saber o que fazer
Se vai me julgar mais uma vez
Não pergunte coisas que eu não sei
Coisas que eu não sei

Não sabemos onde vai parar
Sei que até você vai duvidar
Mas não vou jurar nem prometer
Algo que não sei se vou fazer
Eu não sei se é eterno
Mas posso te dar todo meu tempo

Já não sei mais nada
Já não sei mais nada
Se estaremos juntos
Até o fim do mundo
Eu não sei se sou pra você
Se você é pra mim

O amor que amamos sonhando
Já não sei mais nada
Já não sei mais nada
Se vai ser assim

Seu café derrama no sofá
Distraída não sabe o que faz
Não consegue ver que acabou
Não quer desistir mais deste amor
Mas tudo que vê é o que sou
Não me peça mais do que eu te dou
Não!

Já não sei mais nada
Já não sei mais nada
Se estaremos juntos
Até o fim do mundo
Eu não sei se sou pra você
Se você é pra mim
O amor que amamos sonhando
Já não sei mais nada
Já não sei mais nada
Se vai ser assim

Essa vida é como um livro
Cada página é um dia vivido
Que não podemos escrever e apagar
Essa noite eu preciso
Eu preciso tanto te beijar!"""


texto = texto.lower().split("\n")

texto_tok = []
for verso in texto:
    tokens = nltk.word_tokenize(verso, language="portuguese")
    texto_tok.append(tokens)
    
ngramas = 2
k = 0.1
ngramas_pad, vocab = padded_everygram_pipeline(ngramas, texto_tok)
lm = Lidstone(order=ngramas, gamma=k)
lm.fit(ngramas_pad, vocab)



frase = lm.generate(30)
x = slice(0, 10)

ngramas_teste = flatten([list(w) for w in ngramas_pad])
perplexity = lm.perplexity(ngramas_teste)