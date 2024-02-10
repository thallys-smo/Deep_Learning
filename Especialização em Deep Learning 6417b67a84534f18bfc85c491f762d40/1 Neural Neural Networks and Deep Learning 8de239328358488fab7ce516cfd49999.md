# 1. Neural Neural Networks and Deep Learning

# 1.1. Introdução

---

31 de janeiro de 2024 09:00 Escrever uma introdução depois do curso de neural networks

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Notas de Aulas

[C1_W1.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/C1_W1.pdf)

</aside>

# 1.2. Básico da Programação de uma Rede Neural

---

A ideia deste módulo é fornecer

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Notas de Aulas

[C1_W2.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/C1_W2.pdf)

[1. Standard notations for Deep Learning.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/1._Standard_notations_for_Deep_Learning.pdf)

[2. Binary_Classification.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/2._Binary_Classification.pdf)

[3. Logistic_Regression.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/3._Logistic_Regression.pdf)

[4. Logistic_Regression_Cost_Function.pdf](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/4._Logistic_Regression_Cost_Function.pdf)

</aside>

## Regressão Logística como uma Rede Neural

### Classificação Binária

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled.png)

### Regressão Logística

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%201.png)

### Função Custo de Regressão Logística

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%202.png)

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Qual é a diferença entre a função custo (cust function) e a função perda (loss function) na regressão logística?

**Resposta:** A função perda computa o erro para um exemplo de treino, enquanto a função custo é a média da função perda para todo o conjunto de treinamento. 

</aside>

### Descida de Gradiente

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%203.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%204.png)

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Verdadeiro ou Falso. **Uma função convexa sempre tem vários ótimos locais?**

**Resposta: Falso.** 

</aside>

### Derivativos

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%205.png)

### Mais exemplos de Derivativos

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%206.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%207.png)

### Gráficos de Computação

Computar uma função $J(a,b,c)=3 \cdot (a + b \cdot c)$.

- Primeiro computamos: $u = b \cdot  c$
- Depois computamos: $v = a + u$
- Por fim, obtemos $J = 3 \cdot v$

Ao lado, temos a representação gráfica da computação da função.  

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> O gráfico computacional organiza o sentido de computacional via as linhas azuis.

</aside>

```mermaid
flowchart LR
a ---> U
b ---> U
c ---> U
a ---> V
U(u = b.c) ---> V(v= a+u)
V ---> J(J = 3v)

```

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> **Teste:** One step of _______ propagation on a computation graph yields derivative of final output variable. **Resposta:** Backward

</aside>

Uma etapa de propagação para trás em um gráfico de computação produz a derivada da variável de saída final. 

### Derivativos com um gráfico de Computação

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%208.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%209.png)

### Regressão Logistica Gradient Descent

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2010.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2011.png)

### Gradient Descent em M Exemplos

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2012.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2013.png)

## Python e Vetorização

### Vetorização

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> O que é vetorização? **Resposta:** Um vetor em Python é uma estrutura de dados que armazena uma coleção ordenada de elementos. Cada elemento é identificado por um índice, que começa em 0 para o primeiro elemento, 1 para o segundo e assim por diante.

</aside>

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2014.png)

$$

$$

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Não Vetorizado

```python
z = 0
for i in range(n-x):
	z+ = w[i]*x[i]
z+=b
```

</aside>

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Vetorizado

```python
import numpy as np

z = np.dot(w,x) + b

```

</aside>

- Implementação:

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2015.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2016.png)

A partir da imagem acima, percebemos que a versão vetorizada executa muito mais rápido que a versão não vetorizada. 

<aside>
<img src="https://www.notion.so/icons/verified_gray.svg" alt="https://www.notion.so/icons/verified_gray.svg" width="40px" /> Observação

</aside>

Nesse caso, executamos o código numa CPU ((Central Processing Units, unidades de processamento central), mas também é possível executar numa GPUs (Graphics Processing Units, unidades de processamento gráfico), na qual executamos uma SIMD (single instruction multiple data). 

### Mais exemplos de Vetorização

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2017.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2018.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2019.png)

### Vetorização da Regressão Logística

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2020.png)

### Vetorização da Saída de Gradiente da Regressão Logística

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2021.png)

### Transmissão em Python

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2022.png)

```python
cal = A.sum(axis=0) #Soma vertical 
cal = A.sum(axis=1) #Soma horizontal
```

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2023.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2024.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2025.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2026.png)

### Uma observação sobre vetores Python/Numpy

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2027.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2028.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2029.png)

```python
import numpy as no

a = np.random.randn(5) # Cria um vetor com 5 números gausianos aleatórios.

print (a)

print(a.shape)
```

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2030.png)

### Tour Rápido pelos notebooks Jupyter/iPython

### Explicação da Função Custo de Regressão

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2031.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2032.png)

![Untitled](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Untitled%2033.png)

[Teste](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Teste%20ba22c247a5fd4a2aba2f46c55742a129.md)

[Teste](1%20Neural%20Neural%20Networks%20and%20Deep%20Learning%208de239328358488fab7ce516cfd49999/Teste%200cd3679c867d45a995e4b46f7fe63eb3.md)

# 1.3. **Redes Neurais de uma Camada Oculta**(One hidden layer Neural Networks)

## Rede neural Rasa

### Visão Geral das Redes Neurais

### Representação de Redes Neurais

### Computação da Saída de uma Rede Neural

### Vetorização em Vários Exemplos

### Explicação para a implementação Vetorizada

### Funções de Ativação

### Porque o senhor precisa de funções de ativações não lineares?

### Derivados de Funções de Ativações

### Gradient Descend for Neural Netwoks (Descida de Gradiente para Redes Neurais)

### Intuição de Retropropagação

### Iniciazação Aleatória

# 1.4. Redes de Apreendizagens Profundas (Deep Neural Networks)