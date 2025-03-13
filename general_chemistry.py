import numpy as np
import math
import matplotlib.pyplot as plt

def proton_to_pH(H):
    #H = H
    return  - np.log10(H)

def pH_to_proton(pH):
    return 10**-pH


def first_order_reaction(A, B, k_AB ,t=1):
    fresh_B = A * k_AB * t
    B = B + fresh_B
    A = A - fresh_B
    return A , B

def second_order_reaction(A, B, C, k_AB_C ,t=1):
    fresh_C = A*B * k_AB_C * t
    if fresh_C < 0:
        fresh_C = 0
    B = B - fresh_C
    A = A - fresh_C
    C = C + fresh_C
    return A , B, C

def pKa_to_conc(pKa, pH, total):
    r =  10** (pH - pKa)

    """"
    pH = pKa + log10(A/AH)
    r = A/AH
    and r =  10 ^ (pH - pKa)
    
    A = x * AH 
    AH = tot - A
    A = x * (tot - A)
    A = x*tot - x*A
    A + x*A = x*tot
    (1+x) = x*tot / A
    (1+x) / (x*tot) = 1/A
    A = (x*tot) / (1+x)
    """

    A = (r*total) / (1+r)
    AH = total - A

    return A , AH

def pKa_to_conc_diprotic(pKa, pKa2, pH, total):
    r =  10** (pH - pKa)
    r2 = 10** (pH - pKa2)

    """"
    pH = pKa + log10(A/AH)
    r = A/AH
    and r =  10 ^ (pH - pKa)
    
    A = x * AH 
    AH = tot - A
    A = x * (tot - A)
    A = x*tot - x*A
    A + x*A = x*tot
    (1+x) = x*tot / A
    (1+x) / (x*tot) = 1/A
    A = (x*tot) / (1+x)
    """

    A = (r*total) / (1+r)
    AHx = (r2*total) / (1+r2)
    AH2 =  total - AHx
    AH = total - A - AH2


    return A , AH , AH2
   

def acid_graph(pKa1, pKa2, pH, total):
    l = []
    for pH in np.arange(0.1,14, 0.1):
        A , AH, AH2= pKa_to_conc_diprotic(7, 5, pH, total)

        l.append([pH, A , AH, AH2])

    l = np.array(l)

    fig, ax = plt.subplots()     
    ax.plot(l[:,0], l[:,1], label="A-")
    ax.plot(l[:,0], l[:,2], label="HA")
    ax.plot(l[:,0], l[:,3], label="H2A")
    ax.legend()

def get_eq_point(reaction , equilibrium_point, tolerance, AB):
    B_init = reaction[0][2]
    
    if B_init < equilibrium_point:
        for data_point in AB:
            if data_point[2] > equilibrium_point - equilibrium_point*tolerance:
                return data_point
        
    if B_init > equilibrium_point:
        for data_point in AB:
            if data_point[2] < equilibrium_point + equilibrium_point*(1-tolerance):
                return data_point

    '''
    print(f"final values:\n{A_name}: {reaction[-1][1]:.2f}\n{B_name}: {reaction[-1][2]:.2f}\nk = {k_reaction}")
    tolerance = 0.01
    eq_time = get_eq_point(reaction, equilibrium_point, tolerance)
    print(f"Reaction reach equilibrium within {tolerance*100}% at {eq_time[2]:.2f}s")
    '''