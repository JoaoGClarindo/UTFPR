import numpy as np

def solve_nozzle(algo='SIMPLE', alpha_u=0.5, alpha_p=0.8, max_iter=100, tol=1e-6):
    # Parâmetros Geométricos e Físicos
    rho = 1.0
    AA = 3.0
    AB = 1.0
    p1 = 28.0
    p3 = 0.0

    # Condições Iniciais
    uA = 5.0 / 3.0
    uB = 5.0
    p2 = 25.0

    # Histórico para gráficos
    history = []

    print(f"\n--- Algoritmo {algo} (alpha_u={alpha_u}, alpha_p={alpha_p}) ---")
    print(f"{'Iter':<5} {'uA':<10} {'uB':<10} {'p2':<10} {'Mass Res':<10}")

    for it in range(1, max_iter + 1):
        # 1. Atualizar coeficientes (F = rho * u * A)
        # Usamos valores da iteração anterior (*)
        FA = rho * uA * AA
        
        FB = rho * uB * AB

        # Coeficientes do Momentum (aP * uP = sum(anb * unb) + Source)
        # uA: aA * uA = (p1 - p2)*AA
        aA_mom = FA
        srcA = (p1 - p2) * AA
        
        # uB: aB * uB = aW * uA + (p2 - p3)*AB
        # Nota: aW é o fluxo que vem de A. aB é o fluxo que sai em B.
        aB_mom = FB
        aW_mom = FA # Upwind de A para B
        srcB = (p2 - p3) * AB

        # SIMPLER: Passo extra para calcular Pseudo-Velocidades (u_hat) e Pressure Equation
        if algo == 'SIMPLER':
            # Pseudo-velocidades (sem gradiente de pressão)
            # uA_hat = (0) / aA -> Neste problema específico, uA_hat seria 0 se tirarmos a pressão?
            # A definição é u_hat = (sum anb unb + Source_no_press) / aP
            uA_hat = 0 # Não há vizinho nem fonte não-pressão para uA
            uB_hat = (aW_mom * uA) / aB_mom 
            
            # Coeficientes para Equação de Pressão (NÃO é a correção p')
            # A eq de pressão tem a mesma forma da eq de p', mas usa u_hat
            # aP_P * p2 = aE*p3 + aW*p1 + b_P
            dA_Simpler = AA / aA_mom
            dB_Simpler = AB / aB_mom # SIMPLER usa d do SIMPLE
            
            aP_press = rho * AA * dA_Simpler + rho * AB * dB_Simpler
            # Fonte b_P baseada nas pseudo-velocidades
            # b_P = rho*AA*uA_hat - rho*AB*uB_hat
            b_P = rho * AA * uA_hat - rho * AB * uB_hat
            
            # Termos vizinhos (p1 e p3 são fixos, entram na fonte)
            # u = u_hat - d * grad(p)
            # Continuidade: rho*A*(u_hat - d*grad(p)) ...
            # p2 = (rho*AA*dA*p1 + rho*AB*dB*p3 + b_P) / aP_press
            p2_new = (rho * AA * dA_Simpler * p1 + rho * AB * dB_Simpler * p3 + b_P) / aP_press
            p2 = p2_new # Atualiza p2 para usar no momentum

            # Recalcula coeficientes de momentum e fontes com o NOVO p2
            srcA = (p1 - p2) * AA
            srcB = (p2 - p3) * AB
        
        # 2. Resolução do Momentum (Cálculo de u*)
        # uA* = Source / aA
        uA_star = srcA / aA_mom
        # uB* = (aW * uA* + Source) / aB (Importante: uA* atualizado ou antigo? Geralmente TDMA usa o mais recente)
        # Aqui usamos o uA da iteração anterior para o coeficiente vizinho, mas podemos usar uA_star se for Gauss-Seidel.
        # Vamos usar uA (antigo) no termo vizinho para manter consistência com formulação padrão explícita nos vizinhos
        uB_star = (aW_mom * uA + srcB) / aB_mom 

        # Sub-relaxação da velocidade (u = alpha * u_new + (1-alpha) * u_old)
        uA_star = alpha_u * uA_star + (1 - alpha_u) * uA
        uB_star = alpha_u * uB_star + (1 - alpha_u) * uB

        # 3. Equação de Correção de Pressão (p')
        # aP * p2' = b
        
        # Definição dos d (coeficientes de velocidade)
        if algo == 'SIMPLE' or algo == 'SIMPLER':
            dA = AA / aA_mom
            dB = AB / aB_mom
        elif algo == 'SIMPLEC':
            # d = A / (aP - sum(anb))
            dA = AA / aA_mom # uA não tem vizinho u
            dB = AB / FB * (alpha_u/(1- alpha_u)) # uB tem vizinho uA

        # Coeficientes da Eq de p'
        # uA = uA* - dA * p2'
        # uB = uB* + dB * p2'
        # Continuidade: rho*AA*uA = rho*AB*uB => rho*AA*(uA* - dA*p2') = rho*AB*(uB* + dB*p2')
        # p2' * (rho*AA*dA + rho*AB*dB) = rho*AA*uA* - rho*AB*uB*
        
        aP_pp = rho * AA * dA + rho * AB * dB
        b_pp = rho * AA * uA_star - rho * AB * uB_star # Resíduo de massa
        
        p2_prime = b_pp / aP_pp

        # 4. Correção
        # Correção das velocidades
        uA_new = uA_star - dA * p2_prime
        uB_new = uB_star + dB * p2_prime

        # Correção da pressão
        if algo == 'SIMPLER':
            # No SIMPLER, p2 já foi calculado. Não corrigimos p com p', apenas velocidades.
            # p' serve apenas para projetar a velocidade no campo de massa conservativo.
            pass 
        else:
            # SIMPLE e SIMPLEC atualizam p
            p2_new = p2 + alpha_p * p2_prime
            p2 = p2_new

        # Atualiza variáveis para próxima iteração
        uA = uA_new
        uB = uB_new
        
        history.append((it, uA, uB, p2, abs(b_pp)))
        print(f"{it:<5} {uA:<10.4f} {uB:<10.4f} {p2:<10.4f} {abs(b_pp):<10.2e}")

        if abs(b_pp) < tol:
            print("Converged!")
            break

    return history

# Executando os casos solicitados
print(">>> Executando SIMPLE...")
solve_nozzle(algo='SIMPLE', alpha_u=0.5, alpha_p=0.8)

print("\n>>> Executando SIMPLEC...")
solve_nozzle(algo='SIMPLEC', alpha_u=0.5, alpha_p=1.0) # SIMPLEC geralmente permite relaxação total

print("\n>>> Executando SIMPLER...")
solve_nozzle(algo='SIMPLER', alpha_u=1, alpha_p=0.5)
