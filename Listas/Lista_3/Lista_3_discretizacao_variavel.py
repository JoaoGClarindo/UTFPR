import numpy as np
import matplotlib.pyplot as plt

def solucao_analitica():
    # --- PARÂMETROS FÍSICOS ---
    L = 3               # Espessura da parede (m)
    k = 30               # Condutividade térmica (W/m.K)
    q_dot = 300          # Geração interna de calor (W/m^3)
    q0_prime2 = 400      # Fluxo de calor em x=0 (W/m^2)
    h = 50                # Coeficiente de convecção em x=L (W/m^2.K)
    T_inf = 20             # Temperatura do fluido ambiente (°C)

    # --- CÁLCULO DO PERFIL ---
    # Criamos 100 pontos de x de 0 até L
    x = np.linspace(0, L, 100)
    
    # Implementação da equação analítica:
    # T(x) = (q_dot/2k)*(L^2 - x^2) + (q0/k)*(L - x) + (q_dot*L + q0)/h + T_inf
    
    termo_geracao = (q_dot / (2 * k)) * (L**2 - x**2)
    termo_fluxo = (q0_prime2 / k) * (L - x)
    termo_conveccao = (q_dot * L + q0_prime2) / h
    
    termo_1 = (q_dot / (2 * k)) 
    termo_2 = (q0_prime2 / k)
    termo_3 = (q_dot * L + q0_prime2) / h

    print(f'Termo Geração: {termo_1}')
    print(f'Termo Fluxo: {termo_2}')
    print(f'Termo Convecção: {termo_3}')

    T = termo_geracao + termo_fluxo + termo_conveccao + T_inf

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(10, 6))
    plt.plot(x * 100, T, color='blue', linewidth=2.5, label='Perfil $T(x)$')
    
    # Estilização do gráfico
    plt.title('Distribuição de Temperatura na Parede (1D Permanente)', fontsize=14)
    plt.xlabel('Posição $x$ (cm)', fontsize=12)
    plt.ylabel('Temperatura $T$ (°C)', fontsize=12)
    
    # Destacando as condições de contorno
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Fluxo em $x=0$')
    plt.axvline(x=L*100, color='green', linestyle='--', alpha=0.5, label='Convecção em $x=L$')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Mostrar valores nas extremidades
    plt.text(0, T[0], f' {T[0]:.1f}°C', color='red', fontweight='bold')
    plt.text(L*100, T[-1], f' {T[-1]:.1f}°C', color='green', fontweight='bold', ha='right')

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

def solucao_numerica(L, n_elementos, nos):

    nos=nos
    A = np.zeros((n_elementos,n_elementos+1))
    b = np.zeros(3)
    delta_x = L / n_elementos
    i=1
    # --- Chamada das suas funções para obter os coeficientes ---
    # Nota: Armazenamos os retornos para preencher a matriz A e o vetor b
    A[0,:] = coeficientes_fronteira_ponto_2(i, delta_x, nos, k, q_dot, q0_prime2)
    for i in range(2, n_elementos):
        A[i-1,:] = coeficientes_ponto_3(i, delta_x, nos, k, q_dot)
    i=n_elementos
    A[n_elementos-1,:] = coeficientes_fronteira_ponto_4(i, delta_x, nos, k, q_dot, h, T_inf)

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

    T1 = T[0] +(q0_prime2 / k) * (delta_x/2)
    Tn = ((k/(delta_x/2)*T[-1]) + (h * T_inf)) / (k/(delta_x/2) + h)
    
    T = np.concatenate(([T1], T, [Tn]))

    print("\n--- Resultados das Temperaturas ---")
    for i in range(n_elementos+2):
        print(f"T{i+1}: {T[i]:.2f} °C")




def coeficientes_fronteira_ponto_2(i, delta_x, nos, k, q_dot, q0_prime2):
    del_x_b = nos[i] - nos[i-1]
    del_x_i = nos[i+1] - nos[i]
    S_P = 0
    S_C = q_dot
    a_E = k/(del_x_i)
    a_W = 0
    a_P = a_E - S_P*delta_x
    S_Ca = q0_prime2/delta_x
    b = (S_Ca + S_C)*delta_x

    coeficientes = np.zeros(np.size(nos)-1)
    coeficientes[i-1] = a_P
    coeficientes[i] = -a_E
    coeficientes[-1] = b

    return coeficientes

def coeficientes_ponto_3(i, delta_x, nos, k, q_dot):
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

def coeficientes_fronteira_ponto_4(i, delta_x, nos, k, q_dot, h, T_inf):
    del_x_b = nos[i+1] - nos[i]
    del_x_w = nos[i] - nos[i-1]
    S_P = 0
    S_C = q_dot
    a_E = 0
    a_W = k/(del_x_w)
    denominator = (1/h + del_x_b/k) * delta_x
    S_CA = T_inf / denominator
    print(f'S_CA: {S_CA}')
    S_PA = -1 / denominator
    print(f'S_PA: {S_PA}')
    a_P = a_W - (S_PA + S_P)*delta_x
    b = (S_CA + S_C)*delta_x
    print(f'Equação 3:  {a_P} * T4 = {a_W} * T3 + {a_E} * T5 + {b}')

    coeficientes = np.zeros(np.size(nos)-1)
    print(coeficientes)

    coeficientes[i-2] = -a_W
    coeficientes[i-1] = a_P
    coeficientes[i] = -a_E
    coeficientes[-1] = b

    return coeficientes
    
if __name__ == "__main__":
    solucao_analitica()

    k = 30                  # Condutividade térmica (W/m.K)
    q_dot = 300             # Geração interna de calor (W/m^3)
    q0_prime2 = 400         # Fluxo de calor em x=0 (W/m^2)
    h = 50                  # Coeficiente de convecção em x=L (W/m^2.K)
    T_inf = 20              # Temperatura do fluido ambiente (°C)
    L = 3                   # Espessura da parede (m)
    n_elementos = 10
    nos, faces = np.zeros(n_elementos+2), np.zeros(n_elementos+1)
    faces, nos = discretizar_dominio_1d(L, n_elementos)

    solucao_numerica(L, n_elementos, nos)
    