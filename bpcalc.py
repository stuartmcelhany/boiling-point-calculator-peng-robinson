import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- Boiling Point Temperature Calculator using Peng Robinson EoS ---

# --- Units ---
# Pressure in MPa
# Temperature in K


# --- Inputs ---
P_final = 0.3       
P_initial = 0.01    
P_increment = 0.005
TOL = 0.001 # tolerance for bisection method

# --- Constants for Water ---
Tc = 647            # K
Pc = 22.064         # MPa  
w = 0.344           # unitless
R = 8.314E-6        # MPa m^3 mol^-1 K^-1

# --- Calculated Constants ---
ac = 0.45724*(R**2)*(Tc**2)/Pc
b = 0.07780*R*Tc/Pc
k = 0.37464+1.54226*w-0.26992*(w**2)

# Reduced Temperature
def Tr(T):
  return T/Tc

# PREOS a constant
def a_T(T):
  return (1+(k*(1-np.sqrt(Tr(T)))))**2

# PREOS A constant for Z form
def A(T,P):
  return ac*a_T(T)*P/((R*T)**2)

# PREOS B constant for Z form
def B(T,P):
  return P*b/(R*T)

# Z^3 - (1-B)Z^2 + (A - 3B^2 -2B)Z - (AB - B^2 - B^3) = 0
#This function takes in a T and P and the real root is returned
def roots(T,P):
  coef = [1, -(1-B(T,P)), (A(T,P)-3*(B(T,P)**2)-2*B(T,P)), -(A(T,P)*B(T,P)-(B(T,P)**2)-(B(T,P)**3))]
  r = np.roots(coef)
  return r

# returns the number of real roots
def num_real_roots(T,P):
    r = roots(T,P)
    real_count = 0
    for i in r:
        if np.isreal(i):
            real_count += 1
    return real_count


# Finds the minimum temperature to achieve 3 real roots
def T_min(T,P):
    while num_real_roots(T, P) < 2:
        T += 1.0
    return T

# Finds the maximum temperature to achieve 3 real roots
def T_max(T,P):
    while num_real_roots(T, P) < 2:
        T -= 1.0
    return T

# Returns the fugacity coefficient
def fugacity(Z,CA,CB):
    p1 = np.log(Z-CB)
    p2 = CA/(CB*np.sqrt(8))
    p3 = Z+(1+np.sqrt(2))*CB
    p4 = Z+(1-np.sqrt(2))*CB
    p5 = Z-1
    return np.exp(-p1-p2*np.log(p3/p4)+p5)

# This is the function to be used in the bisection method
# The fugacity of the saturated vapor phase should equal the fugacity of the saturated liquid phase
def func(T,P):
    Z = roots(T,P)
    CA = A(T,P)
    CB = B(T,P)
    Zv = Z[0]
    Zl = Z[2]
    return fugacity(Zv,CA,CB) - fugacity(Zl,CA,CB)

# Bisection method finds the temperature at which the two fugacities are equal
# This temperature is the boiling point at the given pressure
def bisection(t_min,t_max, P):
    t_min = T_min(t_min,P)
    t_max = T_max(t_max,P)
    if (func(t_min,P)*func(t_max,P) >=0):
        print("Try different left and right guesses...")
        return
    mid = t_min
    while ((t_max - t_min) >= TOL):
        mid = (t_min + t_max)/2
        if (func(mid, P) == 0.0):
            break
        if (func(mid, P)*func(t_min,P) < 0):
            t_max = mid
        else:
            t_min = mid
    return mid


# *--- --- --- Driver Code --- --- ---* #
P_sat = []
T_sat = []

P = P_initial

# Generates values of P and runs them in the bisection function to find the corresponding T
while P <= P_final:
    P_sat.append(P)
    t_min, t_max = 1, 1000
    T = bisection(t_min, t_max, P)
    T_sat.append(T)
    print ("Boiling temp at","%.3f"%P,"MPa is","%.2f"%T, "K")
    P += P_increment
T = bisection(t_min, t_max, P)
T_sat.append(T)
P_sat.append(P)
print ("Boiling temp at","%.3f"%P,"MPa is","%.2f"%T, "K")

# Plotting the above matrices
fig, ax = plt.subplots()
ax.plot(P_sat, T_sat, label='Saturation line')

ax.set(xlabel='Saturation Pressure (MPa)', ylabel='Saturation Temperature (K)',
       title='Boiling Point for Water (Peng-Robinson)')
ax.grid()
plt.show()