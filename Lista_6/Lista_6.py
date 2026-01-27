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

def solucao_numerica():
    L = 3                   # Espessura da parede (m)
    k = 30                  # Condutividade térmica (W/m.K)
    q_dot = 300             # Geração interna de calor (W/m^3)
    q0_prime2 = 400         # Fluxo de calor em x=0 (W/m^2)
    h = 50                  # Coeficiente de convecção em x=L (W/m^2.K)
    T_inf = 20              # Temperatura do fluido ambiente (°C)
    delta_x = 1
    # --- Chamada das suas funções para obter os coeficientes ---
    # Nota: Armazenamos os retornos para preencher a matriz A e o vetor b
    a1_e1, a2_e1, a3_e1, b2 = coeficientes_fronteira_ponto_2(delta_x, k, q_dot, q0_prime2)
    a2_e2, a3_e2, a4_e2, b3 = coeficientes_ponto_3(delta_x, k, q_dot)
    a3_e3, a4_e3, a5_e3, b4 = coeficientes_fronteira_ponto_4(delta_x, k, q_dot, h, T_inf)

    # --- Montagem da Matriz A e Vetor b ---
    # A matriz A deve ser (3x3) para as incógnitas [T2, T3, T4]
    A = np.array([
        [ a2_e1, -a3_e1,      0],  # Equação do Ponto 2
        [-a2_e2,  a3_e2, -a4_e2],  # Equação do Ponto 3
        [     0, -a3_e3,  a4_e3]   # Equação do Ponto 4
    ])

    b = np.array([b2, b3, b4])

    # --- Resolução do Sistema ---
    try:
        T = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Erro: A matriz é singular e não pode ser resolvida.")

    T1 = T[0] +(q0_prime2 / k) * (delta_x/2)
    T5 = ((k/(delta_x/2)*T[2]) + (h * T_inf)) / (k/(delta_x/2) + h)
    

    print("\n--- Resultados das Temperaturas ---")
    print(f"T1: {T1:.2f} °C")
    print(f"T2: {T[0]:.2f} °C")
    print(f"T3: {T[1]:.2f} °C")
    print(f"T4: {T[2]:.2f} °C")
    print(f"T5: {T5:.2f} °C")

    

def coeficientes_fronteira_ponto_2(delta_x, k, q_dot, q0_prime2):
    #a2*T2 = a3*T3 + a1*T1 + b2
    S_P = 0
    S_C = q_dot
    a_3 = k/(delta_x)
    a_1 = 0
    a_2 = a_3 - S_P*delta_x
    S_Ca = q0_prime2/delta_x
    b_2 = (S_Ca + S_C)*delta_x
    print(f'Equação 1:  {a_2} * T2 = {a_3} * T3 + {a_1} * T1 + {b_2}')

    return a_1, a_2, a_3, b_2

def coeficientes_ponto_3(delta_x, k, q_dot):
    #a3*T3 = a4*T4 + a2*T2 + b3
    S_P = 0
    S_C = q_dot
    a_4 = k/delta_x
    a_2 = k/delta_x
    a_3 = a_4 + a_2 - S_P*delta_x
    b_3 = (S_C)*delta_x
    print(f'Equação 2:  {a_3} * T3 = {a_4} * T4 + {a_2} * T2 + {b_3}')

    return a_2, a_3, a_4, b_3

def coeficientes_fronteira_ponto_4(delta_x, k, q_dot, h, T_inf):
    #a4*T4 = a3*T3 + a5*T5 + b4
    S_P = 0
    S_C = q_dot
    a_5 = 0
    a_3 = k/(delta_x)
    del_x = delta_x/2
    denominator = (1/h + del_x/k) * delta_x
    S_CA = T_inf / denominator
    print(f'S_CA: {S_CA}')
    S_PA = -1 / denominator
    print(f'S_PA: {S_PA}')
    a_4 = a_3 - (S_PA + S_P)*delta_x
    b_4 = (S_CA + S_C)*delta_x
    print(f'Equação 3:  {a_4} * T4 = {a_3} * T3 + {a_5} * T5 + {b_4}')

    return a_3, a_4, a_5, b_4

def coeficientes_internos():
    # Esta função pode ser implementada para calcular coeficientes internos
    pass

if __name__ == "__main__":
    solucao_analitica()
    solucao_numerica()
