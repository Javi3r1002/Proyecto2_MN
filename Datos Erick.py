# -*- coding: utf-8 -*-
"""
MÉTODOS NUMÉRICOS, PROYECTO 2: AJUSTE DE CURVAS

Integrantes:
    Erick Alvarez (20900)
    Dieter Loesener (20724)
    Javier Mejía (20304)
    Valeria Paiz (191555)
    Edgar Reyes (20061)
    
"""

"""IMPORTAR LIBRERÍAS"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize  
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy.stats as stats
import pylab as py

"""DEFINIR EL MODELO TEÓRICO (FUNCIÓN SENO AMORTIGUADA)"""
def modelo(x,a,b,w,d,c):
        return a * np.exp(-b * x) * np.sin(w * x + d) + c
    
"""DEFINIR FUNCIÓN QUE HAGA EL AJUSTE DE LA CURVA POR MÍNIMOS CUADRADOS"""
def regresion(x,y):
    params, params_covariance = optimize.curve_fit(modelo, x, y, p0 = [10,3,20,1,-0.1])
    """Dándole los parámetros de la función teórica, la lista de los datos y unas
       aproximaciones iniciales de los parámetros que tiene la función teórica, 
       se procede a trabajrlo como un problema de optimización, variando los parámetros
       de la función y devolviendo la mejor estimación"""
    return params

"""FUNCIÓN PARA GRAFICAR LOS DATOS JUNTO CON EL MODELO"""    
def graficaComp(x,y,a,b,w,phi,c):
    datX = x
    datY = y
    
    params = regresion(datX, datY) #Llamar a la función regresión
    
    amp = params[0] # nombrando los parámetros resultantes de la regresión
    beta = params[1]
    omega = params[2]
    phi = params[3]
    desp = params[4]
    
    argExp = [-beta * z for z in x] # operando para obtener la curva teórica de la regresión
    sinTemp = [omega * t for t in x]
    desfase = [phi * g for g in list(np.ones(len(x)))]
    argSin = [sinTemp[h] + desfase[h] for h in range(len(sinTemp))]
    despL = [desp * k for k in list(np.ones(len(x)))]
    
    M1 = amp * np.exp(argExp)
    M2 = np.sin(argSin)
    M3 = M1 * M2
    
    ya = [M3[l] + despL[l] for l in range(len(M3))] # Y de la regresión
    xa = list(np.linspace(min(x),max(x),len(x))) # Lista de valores en X para graficar
    
    promY = sum(datY)/len(datY) # calculando el coeficiente de determinación
    r2num = sum([(d - promY)**2 for d in ya])
    r2den = sum([(s - promY)**2 for s in datY])
    r2 = r2num/r2den
    
    
    plt.figure(figsize=(6, 4)) # mostrando la gráfica de los datos y la curva teórica dada por la regresión
    plt.scatter(x, y, label='Datos')
    plt.plot(xa,ya, label = 'Ajuste')
    plt.ylabel('Aceleración (m/s^2)')
    plt.xlabel('Tiempo (s)')
    plt.figtext(0.225, 0.8,"y = " + str(round(a,3)) + "*exp(-" + str(round(b,3)) + "*x)*sin(" + str(round(w,3)) + "*t + " + str(round(phi,3)) + ") + " + str(round(c,3)) + '\n R^2 = ' +  str(round(r2,3)))
    plt.show()


    
"""FUNCIÓN PARA LECTURA DE DATOS PARA CADA NIVEL"""
def leerNivel1():
    X = []
    Y = []
    
    n1 = pd.read_csv("1.csv", sep = ",", header = None, names = ["x", "y"]) # leer el archivo y almacenarlo en un dataFrame
    
    for i in range(len(n1)): # recorrer el dataFrame
        if n1.loc[i, "x"]>30.6: # reconocer desde dónde los datos mostraban el fenómeno estudiado
            X.append(n1.loc[i, "x"]) # almacenar los valores en istas
            Y.append(n1.loc[i, "y"])
            
    Xsi = [a - 29 for a in X] # ajustar los datos leídos para que la regresión no se pierda con los parámetros iniciales que se le dieron
    return Xsi, Y #Devolver listas con los datos leídos y ajustados
    
    
  
def leerNivel2(): 
    X = []
    Y = []
    
    n1 = pd.read_csv("2.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>36.05:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
            
    Xsi = [a - 35 for a in X]
    return Xsi, Y
    

def leerNivel3():
    X = []
    Y = []
    
    n1 = pd.read_csv("3.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>42.2:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
            
    Xsi = [a - 35 for a in X]
    return Xsi, Y

    
def leerNivel4():
    X = []
    Y = []
    
    n1 = pd.read_csv("4.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>47.6:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
            
    Xsi = [a - 46 for a in X]
    return Xsi, Y
    

def leerNivel5():
    X = []
    Y = []
    
    n1 = pd.read_csv("5.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>60.9:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
            
    Xsi = [a - 60 for a in X]
    return Xsi, Y
    

def leerNivel6():
    X = []
    Y = []
    
    n1 = pd.read_csv("6.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>66.49:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    
    Xsi = [a - 66 for a in X]
    return Xsi, Y
    
    
def leerNivel7():
    X = []
    Y = []
    
    n1 = pd.read_csv("7.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>72.4:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    
    Xsi = [a - 71 for a in X]
    return Xsi, Y
    
    
def leerSot1():
    X = []
    Y = []
    
    n1 = pd.read_csv("s1.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>25.35:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    
    Xsi = [a - 24 for a in X]
    return Xsi, Y
    
    
def leerSot2():
    X = []
    Y = []
    
    n1 = pd.read_csv("s2.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>19.2:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    
    Xsi = [a - 18 for a in X]
    return Xsi, Y
    
    
def leerSot3():
    X = []
    Y = []
    
    n1 = pd.read_csv("s3.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>13.11:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    
    Xsi = [a - 12 for a in X]
    return Xsi, Y

def leerSot4():
    X = []
    Y = []
    
    n1 = pd.read_csv("s4.csv", sep = ",", header = None, names = ["x", "y"])
    
    for i in range(len(n1)):
        if n1.loc[i, "x"]>5.68:
            X.append(n1.loc[i, "x"])
            Y.append(n1.loc[i, "y"])
    return X, Y


"""FUNCIÓN PARA MOSTRAR LA GRÁFICA DE LOS DATOS DE CADA NIVEL Y SU REGRESIÓN"""    
def grafSot4():
    listas = leerSot4() # leer los datos
    params = regresion(listas[0], listas[1]) # darle a la función regresión, los datos leídos
    a = round(params[0],3) # nombrar los parámetros ajustados dados por la regresión
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c) # graficar
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafSot3():
    listas = leerSot3()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafSot2():
    listas = leerSot2()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))

    
def grafSot1():
    listas = leerSot1()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
        
def grafNiv1():
    listas = leerNivel1()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv2():
    listas = leerNivel2()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv3():
    listas = leerNivel3()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv4():
    listas = leerNivel4()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv5():
    listas = leerNivel5()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv6():
    listas = leerNivel6()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    
def grafNiv7():
    listas = leerNivel7()
    params = regresion(listas[0], listas[1])
    a = round(params[0],3)
    b = round(params[1],3)
    w = round(params[2],3)
    phi = round(params[3],3)
    c = round(params[4],3)
    graficaComp(listas[0], listas[1], a, b, w, phi, c)
    print("Regresión: y = " + str(a) + "*exp(-" + str(b) + "*x)*sin(" + str(w) + "*t + " + str(phi) + ") + " + str(c))
    

"""FUNCIÓN PARA ENCONTRAR LA RELACIÓN ENTRE EL CUADRADO DEL PERIODO Y EL NIVEL DEL EDIFICIO"""
def periodo():
    w = [] # lista para almacenar la velocidad angular
    
    w.append(regresion(leerSot4()[0], leerSot4()[1])[2]) # guardar en la lista el valor del parámetro ajustado de cada nivel
    w.append(regresion(leerSot3()[0], leerSot3()[1])[2])
    w.append(regresion(leerSot2()[0], leerSot2()[1])[2])
    w.append(regresion(leerSot1()[0], leerSot1()[1])[2])
    w.append(regresion(leerNivel1()[0], leerNivel1()[1])[2])
    w.append(regresion(leerNivel2()[0], leerNivel2()[1])[2])
    w.append(regresion(leerNivel3()[0], leerNivel3()[1])[2])
    w.append(regresion(leerNivel4()[0], leerNivel4()[1])[2])
    w.append(regresion(leerNivel5()[0], leerNivel5()[1])[2])
    w.append(regresion(leerNivel6()[0], leerNivel6()[1])[2])
    w.append(regresion(leerNivel7()[0], leerNivel7()[1])[2])
    
    
    T2 = [(4*np.pi*np.pi)/(x*x) for x in w] # calcular el periodo a partir de la velocidad angular y almacenarlo en una lista
    x = list(np.linspace(len(T2),0,len(T2)))
        
    regresion_lineal = LinearRegression() # calcular la regresión lineal para comprobar la relación
    regresion_lineal.fit(np.array(x).reshape(-1,1), T2) 
    
    r2 = regresion_lineal.score(np.array(x).reshape(-1,1), T2) # calcular el coeficiente de determinación
    
    m = regresion_lineal.coef_ # Nombrar los parámetros dados por el ajuste de la curva
    b = regresion_lineal.intercept_
    
    #print("Regresión: y = " + str(round(m[0],5)) + "*x + " + str(round(b,5)))
    #print("R^2 = " + str(round(r2,3)))
    #print(w)
    """
    xnew = list(np.linspace(0,11,100))
    ynew = [m * x + b for x in xnew]
        
    plt.scatter(x,T2) # graficar
    plt.plot(xnew,ynew)
    plt.ylabel('T^2 (s)')
    plt.xlabel('Nivel')
    plt.figtext(0.15, 0.8,"y = " + str(round(m[0],5)) + "*x + " + str(round(b,5)) + '\n R^2 = ' +  str(round(r2,3)))"""
    return m,b,T2,x,r2

def periodoGraf():
    m = periodo()[0]
    b = periodo()[1]
    T2 = periodo()[2]
    x = periodo()[3]
    r2 = periodo()[4]
    
    xnew = list(np.linspace(0,11,100))
    ynew = [m * x + b for x in xnew]
        
    plt.scatter(x,T2) # graficar
    plt.plot(xnew,ynew)
    plt.ylabel('T^2 (s)')
    plt.xlabel('Nivel')
    plt.figtext(0.15, 0.8,"y = " + str(round(m[0],5)) + "*x + " + str(round(b,5)) + '\n R^2 = ' +  str(round(r2,3)))
    
def residuos():
    m = periodo()[0]
    b = periodo()[1]
    exp = periodo()[2]
    teo = []
    a = 0
    for i in list(range(1,12)):
        a = m*i + b
        teo.append(float(a[0]))
        
    teoR = teo[::-1]
    res = [teoR[k]-exp[k] for k in range(11)]
    
    #print(exp)
    #print(teoR)

    plt.scatter(teo,res)
    plt.ylabel('Teórico - Experimental (s^2)')
    plt.xlabel('Teórico (s^2)')
    plt.show()
    sns.distplot(res);
    plt.title("Residuos de la regresión realizada para determinar el periodo de oscilación de los elvadores del cit")
    plt.ylabel("Tiempo (s)")
    plt.show()
    a,b = stats.probplot(res, dist="norm", plot=py)
    stats.probplot(res, dist="norm", plot=py)
    R = b[2]
    plt.text(-1.5,.01,'Coeficiente de determinación %f'%b[2])
    py.show()
    plt.boxplot(res)
    plt.show()





"""LLAMAR FUNCIONES"""

residuos()
#periodoGraf()
#grafSot1()
#grafSot2()
#grafSot3()
#grafSot4()
#grafNiv1()
#grafNiv2()
#grafNiv3()
#grafNiv4()
#grafNiv5()
#grafNiv6()
#grafNiv7()