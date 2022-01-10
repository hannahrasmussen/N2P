#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from DES import Differential_Equation_Solver as DES


# In[2]:


alpha = 1/137 
btop_ratio = 6.11*10**-10
D = 1  
dmnp = 1.29332 
f_pi = 131 
gA = 1.27 
g_si = 11/2 
g_sf = 2 
Gf = 1.166*10**-11 
hbar = 6.582*10**-22 
me = .511 
mpi_charged = 139.569 
mpi_neutral = 135  
mPL = 1.124*10**22 
mu = 105.661       
x_values, w_values = np.polynomial.laguerre.laggauss(10)

        
# In[3]:


@nb.jit(nopython=True)
def f_elec(Ee,T,eta): 
    return 1/(np.e**((Ee/T)-eta)+1)

@nb.jit(nopython=True)
def f_pos(Ee,T,eta):
    return 1/(np.e**((Ee/T)+eta)+1)

@nb.jit(nopython=True)
def F(Ti,Tf,ai,af): 
    dil_fact = (g_sf*(Tf**3)*(af**3))/(g_si*(Ti**3)*(ai**3))
    return dil_fact

@nb.jit(nopython=True)
def N_eff(T_i,T_f,a_i,a_f,f_final,e_a,bxsz): 
    TCM = T_i*a_i/a_f 
    e_dens = (TCM**4/(2*np.pi**2))*oldtrapezoid(f_final*e_a**3,bxsz)
    neff = 6*e_dens/((7/4)*(4/11)**(4/3)*(np.pi**2/30)*(T_f)**4) 
    return neff 

@nb.jit(nopython=True)
def Yn(T):
    return np.e**(-dmnp/T)/(np.e**(-dmnp/T)+1)

@nb.jit(nopython=True)
def Yp(T):
    return 1/(np.e**(-dmnp/T)+1)


# In[4]:


@nb.jit(nopython=True)
def I1(eps,x): #Energy Density
    numerator = (np.e**eps)*(eps**2)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return numerator/denominator

@nb.jit(nopython=True)
def I2(eps,x): #Pressure
    numerator = (np.e**eps)*(eps**4)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return numerator/denominator

@nb.jit(nopython=True)
def dI1(eps,x): #Derivative of Energy Density
    numerator = (np.e**eps)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return (-x)*numerator/denominator

@nb.jit(nopython=True)
def dI2(eps,x): #Derivative of Pressure
    numerator = (np.e**eps)*3*(eps**2)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return (-x)*numerator/denominator

@nb.jit(nopython=True)
def calculate_integral(I,x):
    return np.sum(w_values*I(x_values,x))  

@nb.jit(nopython=True)
def trapezoid(array,dx,x0,xf,a,B): 
    total = np.sum(dx*(array[1:-2]+array[2:-1])/2)
    
    if (len(array)==1):
        total += (B-a)*array[0]
    
    else:
        array_x0 = (((array[1]-array[0])/dx)*a) + array[0]-(x0*((array[1]-array[0])/dx)) 
        array_xf = (((array[-1]-array[-2])/dx)*B) + array[-1]-(xf*((array[-1]-array[-2])/dx))
    
        diff_start = (x0+dx)-a
        diff_end = B-(xf-dx)
        total += (diff_start*(array_x0+array[1]))/2
        total += (diff_end*(array_xf+array[-2])/2)
    
    return total

@nb.jit(nopython=True)
def newtrapezoid(array,x):
    total = np.sum((x[1:]-x[:-1])*(array[1:]+array[:-1])/2)
    return total

@nb.jit(nopython=True)
def oldtrapezoid(array,dx): 
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total

@nb.jit(nopython=True)
def set_arrs(a,e_arr,f):
    if len(np.where(e_arr==0)[0])>1:
        e_hold = e_arr[:np.where(e_arr==0)[0][1]]
        f = f[:np.where(e_arr==0)[0][1]]
        return e_hold, f
    return e_arr, f


# In[5]:


@nb.jit(nopython=True)
def rate1(ms,mixangle): 
    numerator = 9*(Gf**2)*alpha*(ms**5)*((np.sin(mixangle))**2)
    denominator = 512*np.pi**4
    Gamma = numerator/denominator
    return Gamma

@nb.jit(nopython=True)
def rate2(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(mixangle))**2
    Gamma = part1*part2
    return Gamma

@nb.jit(nopython=True)
def rate3(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma

@nb.jit(nopython=True)
def rate4(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma 

@nb.jit(nopython=True)
def ts(ms,angle):
    return 1/(rate1(ms,angle)+rate2(ms,angle)+rate3(ms,angle)+rate4(ms,angle))

@nb.jit(nopython=True)
def ns(Tcm,t,ms,angle):
    part1 = D*3*1.20206/(2*np.pi**2)
    part2 = Tcm**3*np.e**(-t/ts(ms,angle))
    n_s = part1*part2
    return n_s


# In[6]:


@nb.jit(nopython=True)
def varlam_nue_n(a, e_arr, eta, f, T):
    p_arr = e_arr/a
    f_arr = f
    part1 = 1
    part2 = (p_arr**2)*(p_arr+dmnp)*((p_arr+dmnp)**2 - me**2)**(1/2)
    part3 = f_arr * (1 - f_elec(p_arr+dmnp,T,eta))
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_pos_n(a, e_arr, eta, f, T):
    p_temp = e_arr/a
    if p_temp[-1] < dmnp+me:
        return 0
    p_arr = p_temp[np.where(p_temp > dmnp+me)[0]]
    f_arr = f[np.where(p_temp > dmnp+me)[0]]
    
    part1 = 1
    part2 = (p_arr**2)*(p_arr-dmnp)*((p_arr-dmnp)**2 - me**2)**(1/2)
    part3 = (1-f_arr) * f_pos(p_arr-dmnp,T,eta)
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_n(a, e_arr, eta, f, T):
    p_temp = e_arr/a
    if p_temp[1] > dmnp-me:
        return 0
    p_arr = p_temp[np.where(p_temp < dmnp-me)[0]]
    f_arr = f[np.where(p_temp < dmnp-me)[0]]
    
    part1 = 1
    part2 = (p_arr**2)*(dmnp-p_arr)*((dmnp-p_arr)**2 - me**2)**(1/2)
    part3 = (1-f_arr) * (1-f_elec(dmnp-p_arr,T,eta))
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return 0 #(Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_p_elec(a, e_arr, eta, f, T):
    p_arr = e_arr/a
    f_arr = f
    part1 = 1
    part2 = (p_arr**2)*(p_arr+dmnp)*((p_arr+dmnp)**2 - me**2)**(1/2)
    part3 = (1-f_arr) * f_elec(p_arr+dmnp,T,eta)
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_anue_p(a, e_arr, eta, f, T):
    p_temp = e_arr/a
    if p_temp[-1] < dmnp+me:
        return 0
    p_arr = p_temp[np.where(p_temp > dmnp+me)[0]]
    f_arr = f[np.where(p_temp > dmnp+me)[0]]
    
    part1 = 1
    part2 = (p_arr**2)*(p_arr-dmnp)*((p_arr-dmnp)**2 - me**2)**(1/2)
    part3 = (f_arr) * (1-f_pos(p_arr-dmnp,T,eta))
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_anue_elec_p(a, e_arr, eta, f, T):
    p_temp = e_arr/a
    if p_temp[1] > dmnp-me:
        return 0
    p_arr = p_temp[np.where(p_temp < dmnp-me)[0]]
    f_arr = f[np.where(p_temp < dmnp-me)[0]]
    
    part1 = 1
    part2 = (p_arr**2)*(dmnp-p_arr)*((dmnp-p_arr)**2 - me**2)**(1/2)
    part3 = (f_arr) * (f_elec(dmnp-p_arr,T,eta))
    integrand = part1 * part2 * part3
    integral = newtrapezoid(integrand, p_arr)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)

@nb.jit(nopython=True)
def varlam_np(a, e_arr, eta, f, T):
    return varlam_nue_n(a, e_arr, eta, f, T) + varlam_pos_n(a, e_arr, eta, f, T) + varlam_n(a, e_arr, eta, f, T)

@nb.jit(nopython=True)
def varlam_pn(a, e_arr, eta, f, T):
    return varlam_p_elec(a, e_arr, eta, f, T) + varlam_anue_p(a, e_arr, eta, f, T) + varlam_anue_elec_p(a, e_arr, eta, f, T)

@nb.jit(nopython=True)
def dYn_dt(a,e_arr,eta,f,T,Yn,Yp): 
    n_to_p = varlam_np(a, e_arr, eta, f, T)
    p_to_n = varlam_pn(a, e_arr, eta, f, T)
    return -Yn*n_to_p + Yp*p_to_n

@nb.jit(nopython=True)
def dYp_dt(a,e_arr,eta,f,T,Yn,Yp): 
    n_to_p = varlam_np(a, e_arr, eta, f, T)
    p_to_n = varlam_pn(a, e_arr, eta, f, T)
    return Yn*n_to_p - Yp*p_to_n

#@nb.jit(nopython=True)
def driver(a_arr, e_mat, f_mat, T_arr, t_arr, ms, mixangle):
    Tcm_arr = 1/a_arr
    n_to_p = np.zeros(len(a_arr))
    p_to_n = np.zeros(len(a_arr))  
    H = np.zeros(len(T_arr))
    eta_arr = np.zeros(len(a_arr))

    for i in range(len(a_arr)):
        e_arr, f_arr = set_arrs(a_arr[i],e_mat[i],f_mat[i])
        H_part1 = 8 * np.pi / (3 * mPL**2)
        H_part3 = T_arr[i]**4 * np.pi**2 / 15 
        H_part4 = 2 * T_arr[i]**4 * calculate_integral(I1,me/T_arr[i]) / np.pi**2 
        H_part5 = 0 #7/8 * np.pi**2/30 * 6 * Tcm_arr[i]**4  
        H_part6 = ms*ns(Tcm_arr[i],t_arr[i],ms,mixangle)
        H_part7 = (Tcm_arr[i]**4/(2*np.pi**2))*newtrapezoid(f_arr*e_arr**3,e_arr)
        H[i] = np.sqrt(H_part1 * (H_part3 + H_part4 + H_part5 + H_part6 + H_part7)) / hbar
        n_to_p[i] = varlam_np(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        p_to_n[i] = varlam_pn(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
    
    return n_to_p, p_to_n, H

#@nb.jit(nopython=True)
def individual_driver(a_arr, e_mat, f_mat, T_arr, ms, mixangle):
    nue_n = np.zeros(len(a_arr))
    pos_n = np.zeros(len(a_arr))  
    n = np.zeros(len(a_arr))
    p_elec = np.zeros(len(a_arr))
    anue_p = np.zeros(len(a_arr))
    anue_elec_p = np.zeros(len(a_arr))
    eta_arr = np.zeros(len(a_arr))

    for i in range(len(a_arr)):
        e_arr, f_arr = set_arrs(a_arr[i],e_mat[i],f_mat[i])
        nue_n[i] = varlam_nue_n(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        pos_n[i] = varlam_pos_n(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        n[i] = varlam_n(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        p_elec[i] = varlam_p_elec(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        anue_p[i] = varlam_anue_p(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        anue_elec_p[i] = varlam_anue_elec_p(a_arr[i], e_arr, eta_arr[i], f_arr, T_arr[i])/hbar
        
    return nue_n, pos_n, n, p_elec, anue_p, anue_elec_p

# In[7]:

#@nb.jit(nopython=True)
def YnYp(n2p_arr,p2n_arr,T_arr,t_arr):
    j=10**-10
    for i in range (len(t_arr)-1): 
        if t_arr[i+1]<=t_arr[i]:
            t_arr[i+1] = t_arr[i+1] + j*(1.52*10**21)
            j = j + 10**-10
            
    cs_n2p = CubicSpline(t_arr/(1.52*10**21),n2p_arr)
    cs_p2n = CubicSpline(t_arr/(1.52*10**21),p2n_arr)
    y_in = np.array([Yn(T_arr[0]),Yp(T_arr[0]),Yn(T_arr[0])+Yp(T_arr[0])])
    N = 10000
    dN = 10
    xi = 10**16/(1.52*10**21)
    xf = 10**23/(1.52*10**21)
    dx_init = 10**14/(1.52*10**21)
    
    def ders(x,y):
        dYndt = -y[0]*cs_n2p(x) + y[1]*cs_p2n(x)
        dYpdt = y[0]*cs_n2p(x) - y[1]*cs_p2n(x)
        d_arr = np.array([dYndt,dYpdt,dYndt+dYpdt])
        return d_arr

    x_vals, result, dx = DES.destination_x_dx(ders, y_in, N, dN, xi, xf, dx_init)
    
    return x_vals, result 

@nb.jit(nopython=True)
def entropy_density_photon(x):
    T_local = me/x
    return ((4*np.pi**2)/45)*(T_local**3)

@nb.jit(nopython=True)
def entropy_density_e(x):
    integral_1 = calculate_integral(I1,x)
    integral_2 = calculate_integral(I2,x)
    s = (me/x)**3*(integral_1/np.pi**2 + integral_2/(3*np.pi**2))
    return s

@nb.jit(nopython=True)
def plasma_entropy_density(x):
    return entropy_density_photon(x) + 2*entropy_density_e(x)

@nb.jit(nopython=True)
def spb(a_arr,T_arr): 
    s_arr = np.zeros(len(T_arr))
    nbaryon_arr = np.zeros(len(T_arr))
    spb_arr = np.zeros(len(T_arr)) 
    nbaryon_end = btop_ratio*(2*1.20206/np.pi**2)*T_arr[-1]**3
    
    for i in range(len(T_arr)):
        s_arr[i] = plasma_entropy_density(me/T_arr[i]) 
        nbaryon_arr[i] = nbaryon_end*(a_arr[-1]**3)/a_arr[i]**3
        spb_arr[i] = s_arr[i]/nbaryon_arr[i]
        
    return spb_arr
