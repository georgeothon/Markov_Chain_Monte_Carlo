#Bibliotecas
import numpy as np
import pandas as pd
from scipy.stats import gamma


#Variáveis
C = 1.390000000
R = 1.10300000

def h(x):
    #Retorna um array de booleanos se o elemento pertence ao intervalo
    return np.logical_and( x > 1 , x < 2 )*1

def g(x):
    #Testar com parêmetro lambda
    return gamma.pdf(x,C) * abs(np.cos(R*x))


def Metropolis(n = 100000, dp = 0.5):
    #Executa o método de Metropolis
    
    #Inicia a cadeia e define seu primeiro elemento
    cadeia = []
    x_anterior = abs(np.random.normal(0,dp))        
    prob_anterior = g(x_anterior)
    cadeia.append(x_anterior)
    
    for i in range(1,n):
        
        #Gera um novo "x" e o parâmetro alpha
        x_novo = abs(x_anterior + np.random.normal(0,dp))
        prob_novo = g(x_novo)
        alpha = prob_novo / prob_anterior
        
        #Verifica se X é ou não aceito
        if alpha >= np.random.random() :
            cadeia.append(x_novo)
            x_anterior = x_novo
            prob_anterior = prob_novo
            
        else:
            cadeia.append(x_anterior)
        
        #Calcula o erro padrão, após 2000 iterações
        if i >= 1999:
            erro = np.std(np.array(cadeia))/np.sqrt(len(cadeia))
            
            #Verifica se o erro é menor ou igual a 1%
            if erro <= 0.01:
                return np.array(cadeia)
            
    return np.array(cadeia)


def Baker(n = 100000, dp = 0.5):
    #Executa o método de Metropolis
    
    #Inicia a cadeia e define seu primeiro elemento
    cadeia = []
    x_anterior = abs(np.random.normal(0,dp))        
    prob_anterior = g(x_anterior)
    cadeia.append(x_anterior)

    for i in range(1,n):
        
        #Gera um novo "x" e o parâmetro alpha
        x_novo = abs(x_anterior + np.random.normal(0,dp))
        prob_novo = g(x_novo)
        alpha = prob_novo /( prob_novo + prob_anterior )
        
        #Verifica se X é ou não aceito
        if alpha >= np.random.random() :
            cadeia.append(x_novo)
            x_anterior = x_novo
            prob_anterior = prob_novo
            
        else:
            cadeia.append(x_anterior)
        
        #Calcula o erro padrão, após 2000 iterações
        if i >= 1999:
            erro = np.std(np.array(cadeia))/np.sqrt(len(cadeia))
            
            #Verifica se o erro é menor ou igual a 1%
            if erro <= 0.01:
                return np.array(cadeia)
            
    return np.array(cadeia)

def main():
    
    #Define as colunas da tabela
    colunas = ['Resultado Estimado', 'Nº de Iterações']
    linha, lista1, lista2 = [], [[],[]], [[],[]]
    
    for i in range(10):
        
        #Calcula o resultado estimado da integral com Metropolis
        metropolis = Metropolis()
        lista1[0].append(h(metropolis).mean())
        lista1[1].append(len(metropolis))
    
        #Calcula o resultado estimado da integral
        baker = Baker()
        lista2[0].append(h(baker).mean())
        lista2[1].append(len(baker))
    
    #Calcula a média das respostas e do número de iterações
    metro, bk = np.array(lista1[0]).mean(), np.array(lista2[0]).mean()
    it_metro, it_bk = np.array(lista1[1]).mean(), np.array(lista2[1]).mean()
    
    #Acrescenta as informações na tabela
    linha.append([metro, it_metro])
    linha.append([bk, it_bk])
    #Tranforma a tabela em Data Frame
    df = pd.DataFrame(linha, columns = colunas, index = ['Metropolis', 'Baker'])
    
    print(df)

main()
