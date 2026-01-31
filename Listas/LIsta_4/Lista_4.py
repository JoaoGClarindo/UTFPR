import numpy as np
import matplotlib.pyplot as plt

# A função agora aceita o array numérico como argumento
def solucao_analitica_com_comparacao(T_num_array, L, k, T_0, T_N):
    # --- PARÂMETROS FÍSICOS (ATENÇÃO: Devem ser os mesmos usados no cálculo numérico!) ---
    L = L               # Espessura da parede (m)
    k = k              # Condutividade térmica (W/m.K)
    T_0 = T_0             # Temperatura na face esquerda (°C)
    T_N = T_N             # Temperatura na face direita (°C)

    # --- 1. CÁLCULO DA SOLUÇÃO ANALÍTICA (Linha contínua) ---

    x_analitica = np.linspace(0, L, 100)
    
    T_analitica = (T_N - T_0) * x_analitica / L + T_0

    # --- 2. DEFINIÇÃO DAS COORDENADAS DOS PONTOS NUMÉRICOS ---
    # Assume-se que T_num_array = [T_face_esq, T_centro1, T_centro2..., T_face_dir]
    n_pontos_total = len(T_num_array)
    n_elementos_internos = n_pontos_total - 2
    delta_x_num = L / n_elementos_internos
    
    x_num = []
    # Ponto na face esquerda (x=0)
    x_num.append(0.0)
    # Pontos nos centros dos volumes de controle
    for i in range(n_elementos_internos):
        x_num.append((i + 0.5) * delta_x_num)
    # Ponto na face direita (x=L)
    x_num.append(L)
    
    # Converter para numpy array e para cm (para o gráfico)
    x_num_cm = np.array(x_num) * 100


    # --- 3. VISUALIZAÇÃO ---
    plt.figure(figsize=(10, 6))
    
    # Plot da Linha Analítica (Alterei a cor para preto para destacar os pontos)
    plt.plot(x_analitica * 100, T_analitica, 'k-', linewidth=2, label='Solução Analítica (Exata)')
    
    # --- NOVO: Plot dos Pontos Numéricos ---
    # 'ro' = red circles (círculos vermelhos)
    # markersize = tamanho do ponto
    # markeredgecolor = cor da borda do ponto para contraste
    plt.plot(x_num_cm, T_num_array, 'ro', markersize=9, markeredgecolor='black', label='Solução Numérica (FVM)')

    
    # Estilização do gráfico
    plt.title('Validação: Comparação Numérico vs. Analítico (1D)', fontsize=14)
    plt.xlabel('Posição $x$ (cm)', fontsize=12)
    plt.ylabel('Temperatura $T$ (°C)', fontsize=12)
    
    # Destacando as condições de contorno
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=L*100, color='gray', linestyle='--', alpha=0.5)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def discretizar_dominio_1d(L, n_elementos):
    delta_x = L / n_elementos
    faces = [i * delta_x for i in range(n_elementos + 1)]
    nos = [0] + [(i + 0.5) * delta_x for i in range(n_elementos)] + [L]
    ilustrar_dominio_1d(faces, nos, L, n_elementos, delta_x)
    return faces, nos

def ilustrar_dominio_1d(faces, nos, L, n_elementos, delta_x):
    nomes_nos = [f"T{i+1}" for i in range(n_elementos+2)]

    fig, ax = plt.subplots(figsize=(12, 4))

    # 1. Desenha a linha principal do domínio
    ax.hlines(0, 0, L, colors='black', linewidth=3)

    # 2. Desenha as faces (divisões dos elementos)
    for face in faces:
        ax.vlines(face, -0.1, 0.1, colors='gray', linestyle='--')
    
    # 3. Desenha os nós (pontos de cálculo)
    ax.scatter(nos, [0]*len(nos), color='red', s=100, zorder=5, label='Nós (Pontos de Cálculo)')

    # 4. Adiciona os nomes dos nós (T2, T3, T4)
    for i, no in enumerate(nos):
        ax.text(no, 0.15, nomes_nos[i], ha='center', fontsize=12, fontweight='bold', color='red')

    # --- ANOTAÇÕES TÉCNICAS ---

    # Comprimento Total (L)
    ax.annotate('', xy=(0, -0.4), xytext=(L, -0.4), 
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(L/2, -0.5, f'Comprimento Total (L) = {L} m', ha='center', fontsize=11)

    # Tamanho do Elemento (Delta_x)
    centro_primeiro_elem = delta_x / 2
    ax.annotate('', xy=(0, -0.2), xytext=(delta_x, -0.2), 
                arrowprops=dict(arrowstyle='<->', color='blue'))
    ax.text(delta_x/2, -0.3, f'$\Delta x$ = {delta_x:.4f} m', ha='center', color='blue')

    # Número de Elementos
    ax.text(L/2, 0.6, f'Domínio Discretizado: {n_elementos} Elementos', 
            ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

    # Ajustes finais de exibição
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlim(-0.005, L + 0.005)
    ax.axis('off') # Remove os eixos padrão para parecer um esquema
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def solucao_numerica(L, n_elementos, nos, T_0, T_N, k):

    T_0 = T_0
    T_N = T_N
    k = k
    nos=nos
    A = np.zeros((n_elementos,n_elementos+1))
    b = np.zeros(3)
    delta_x = L / n_elementos
    i=1
    # --- Chamada das suas funções para obter os coeficientes ---
    # Nota: Armazenamos os retornos para preencher a matriz A e o vetor b
    A[0,:] = coeficientes_fronteira_esquerda_Dirichlet(i, delta_x, nos, k, 0, T_0)
    for i in range(2, n_elementos):
        A[i-1,:] = coeficientes_ponto_internos(i, delta_x, nos, k, 0)
    i=n_elementos
    A[n_elementos-1,:] = coeficientes_fronteira_direita_Dirichlet(i, delta_x, nos, k, 0, T_N)
    # --- Montagem da Matriz A e Vetor b ---
    # A matriz A deve ser (3x3) para as incógnitas [T2, T3, T4]

    b = np.array(A[:,-1])
    A = A[0:n_elementos,0:n_elementos]
    print("\nMatriz A:")
    print(A)
    print("\nVetor b:")
    print(b)
    # --- Resolução do Sistema ---
    try:
        T = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Erro: A matriz é singular e não pode ser resolvida.")

    T1 = T_0
    Tn = T_N
    
    T = np.concatenate(([T1], T, [Tn]))

    print("\n--- Resultados das Temperaturas ---")
    for i in range(n_elementos+2):
        print(f"T{i+1}: {T[i]:.2f} °C")

    return T

def coeficientes_fronteira_esquerda_Dirichlet(i, delta_x, nos, k, q_dot, T_0):
    del_x_b = nos[i] - nos[i-1]
    del_x_i = nos[i+1] - nos[i]
    S_P = 0
    S_C = q_dot
    a_E = k/(del_x_i)
    a_W = 0
    a_1 = k/(del_x_b)
    S_Pa = -a_1/delta_x
    #a_P = a_E - (S_Pa + S_P) * delta_x
    a_P = a_E + a_1 - S_P * delta_x
    b = a_1 * T_0 + (S_C)*delta_x
    S_Ca = a_1 * T_0 / delta_x
    #b = (S_Ca + S_C)*delta_x

    coeficientes = np.zeros(np.size(nos)-1)
    coeficientes[i-1] = a_P
    coeficientes[i] = -a_E
    coeficientes[-1] = b

    return coeficientes

def coeficientes_fronteira_direita_Dirichlet(i, delta_x, nos, k, q_dot, T_N):
    del_x_b = nos[i+1] - nos[i]
    del_x_w = nos[i] - nos[i-1]
    S_P = 0
    S_C = q_dot
    a_E = 0
    a_W = k/(del_x_w)
    a_N = k/(del_x_b)
    S_Pa = -a_N/delta_x
    #a_P = a_E - (S_Pa + S_P) * delta_x
    a_P = a_W + a_N - S_P * delta_x
    b = a_N * T_N + (S_C)*delta_x
    S_Ca = a_N * T_N / delta_x
    #b = (S_Ca + S_C)*delta_x

    coeficientes = np.zeros(np.size(nos)-1)
    coeficientes[i-2] = -a_W
    coeficientes[i-1] = a_P
    coeficientes[i] = -a_E
    coeficientes[-1] = b

    return coeficientes

def coeficientes_ponto_internos(i, delta_x, nos, k, q_dot):
    #a3*T3 = a4*T4 + a2*T2 + b3
    del_x_e = nos[i+1] - nos[i]
    del_x_w = nos[i] - nos[i-1]
    S_P = 0
    S_C = q_dot
    a_E = k/del_x_e
    a_W = k/del_x_w
    a_P = a_E + a_W - S_P*delta_x
    b = (S_C)*delta_x

    coeficientes = np.zeros(np.size(nos)-1)
    coeficientes[i-2] = -a_W
    coeficientes[i-1] = a_P
    coeficientes[i] = -a_E
    coeficientes[-1] = b

    return coeficientes

def TDMA_solver(a_W, a_P, a_E, b, n):
    # Inicialização dos vetores
    P = np.zeros(n)
    Q = np.zeros(n)
    T = np.zeros(n)

    # Cálculo dos coeficientes P e Q
    P[0] = a_E[0] / a_P[0]
    Q[0] = b[0] / a_P[0]

    for i in range(1, n):
        denom = a_P[i] - a_W[i] * P[i-1]
        P[i] = a_E[i] / denom
        Q[i] = (b[i] + a_W[i] * Q[i-1]) / denom

    # Back substitution para encontrar T
    T[-1] = Q[-1]
    for i in range(n-2, -1, -1):
        T[i] = P[i] * T[i+1] + Q[i]

    return T



if __name__ == "__main__":

    T_0 = 150               # Temperatura na face esquerda (°C)
    T_N = 50                # Temperatura na face direita (°C)
    k = 25                  # Condutividade térmica (W/m.K)
    h = 50                  # Coeficiente de convecção em x=L (W/m^2.K)
    T_inf = 20              # Temperatura do fluido ambiente (°C)
    L = 1                   # Espessura da parede (m)
    n_elementos = 5
    nos, faces = np.zeros(n_elementos+2), np.zeros(n_elementos+1)
    faces, nos = discretizar_dominio_1d(L, n_elementos)

    T = solucao_numerica(L, n_elementos, nos, T_0, T_N, k)
    solucao_analitica_com_comparacao(T,L, k, T_0, T_N)
    