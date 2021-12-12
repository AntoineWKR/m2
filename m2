# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 07:51:48 2021

@author: antoine
"""

import numpy as np
import matplotlib.pyplot as plt

"""Création de la fonction d'entrée"""

def X(T): #Calcul des coordonnées de X pour la courbe d'entrée
    rep=np.zeros((len(T),1))
    for k in range(len(T)):
        rep[k][0]=2*(np.cos(T[k])+k*np.sin(T[k]))
    return rep

def Y(T): #Calcul des coordonnées de Y pour la courbe d'entrée
    rep=np.zeros((len(T),1))
    for k in range(len(T)):
        rep[k][0]=(2*(np.sin(T[k])-k*np.cos(T[k])))
    return rep    
  
n=30
T=np.linspace(0,2*np.pi,n+1)
t=np.linspace(0,1,n+1)
x=X(T)
y=Y(T)

"""Détermination des poles"""

def factorielle(n): #Calcul le factoriel d'un nombre n
    rep=1
    if int(n)>0:
        for k in range(1,int(n)+1):
            rep=rep*k
    return rep

def k_parmi_n(k,n): #Calcul de k parmi n
    return factorielle(n)/(factorielle(k)*factorielle(n-k))    

def B(i,n,t): #Calcul du coefficient de la base de Bernstein pour un i,n et t donnés
    C=k_parmi_n(i,n)
    B=C*(t**i)*((1-t)**(n-i))      
    return B

def matrice_B(I,n,T): #Calcul de la matrice de Bernstein
    long_T=len(T)
    long_I=len(I)
    B_2=np.zeros((long_T,long_I))
    for k in range(long_T):
        for j in range(long_I):
            B_2[k][j]=B(I[j],n,T[k])
    return B_2   
    
def pole(x,y,B): #Calcul les coordonnées X et Y des poles
    X=np.linalg.solve(B,x)
    Y=np.linalg.solve(B,y)
    return X,Y

"""Algorithme de Casteljau"""

def Casteljau_A(A,i,j,t): #Calcul du coefficient  de Castel jau pour un i,j et t donnés
    if j==1:
        Aij=t*A[i][0]+(1-t)*A[i-1][0]
    else:
        Aij=t*A[i][j-1]+(1-t)*A[i-1][j-1]
    return Aij    

def Casteljau(A,t): #Algorithme de Casteljau
    l=len(A)
    A_2=np.zeros((l,l))  
    A_rep=[]
    #if l==1:
        #return A_2[0]
    for k in range(l):
        Ai0=A[k]
        A_2[k][0]=Ai0
    for k in range(l):
        for i in range(0,l):
            for j in range(1,l):
                A_2[i][j]=Casteljau_A(A_2,i,j,t[k])                
        A_rep.append(A_2[n][n])
    return A_rep


I=np.linspace(0,n,n+1)
mat_B=matrice_B(I, n, t)

Ax,Ay=pole(x,y,mat_B)

Cx=Casteljau(Ax,t)
Cy=Casteljau(Ay,t)

"""Partie graphique"""

plt.figure()
fig, (ax1,ax2)=plt.subplots(1,2)
ax1.plot(x,y,label='entrée',linewidth=0.8)
ax1.plot(Ax,Ay,label='pole',linewidth=0.8)
ax1.legend()
ax2.plot(Ax,Ay,label='pole',linewidth=0.8)
ax2.plot(Cx,Cy,label='sortie Casteljau',linewidth=0.8)
ax2.legend()
plt.show()

            
     
            

    

    
