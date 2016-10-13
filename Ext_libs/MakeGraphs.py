# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:16:47 2016

@author: alex
"""
#cd '/home/alex/Yandex.Disk/Ni63/Отчеты/! Этап 4/06 Ch 12dut/'

import lvm_read as lvm
import numpy as np
import pylab as plt
from scipy.interpolate import interp1d

def parseLvm2UI(folder,files, ext='.lvm'):
    # Просто считываем данные из lvm в (u,i)
    u = []
    i = []
    for item in files:
        if item.find(ext)<0:
            fname = folder + item + ext
        else:
            fname = folder + item
        RawData = lvm.read(fname)
        u.append(RawData[0]['data'][:,0])
        i.append(RawData[0]['data'][:,1])
    return (u,i)

def findPrms(u,i):
    # Ищем Uxx и Iкз
    Uoc = np.array([])
    Isc = np.array([])
    N = len(u)
    for j in range(0,N):
        U = u[j]*1e3 #mV
        I = i[j]*1e9 #nA
        # plt.plot(I, U); plt.grid()
        #ВВодим интерполятор
        fi = interp1d(U, I) #, kind='cubic'
        fu = interp1d(I, U)
        # plt.plot(I, U, I, fu(I)); plt.grid()
        try:
            isc = float(fi(0)) # ток при нулевом напряжении
            uoc = float(fu(0)) # напряжение при нулевом токе
        except Exception:
            isc = 0 #float('NaN')
            uoc = 0 #float('NaN')
        Isc = np.append(Isc,isc)
        Uoc = np.append(Uoc,uoc)
    return (Uoc*1e-3,Isc*1e-9)

# Загружаем данные

folder = '/media/alex/Seagate Expansion Drive/Эксперименты/БЭП/Чистохин 12шт перепайка/25C/'
files = []
for j in range(1,13):
    files.append('chip%i_frw_d' % (j))
(uD,iD) = parseLvm2UI(folder,files)

files = []
for j in range(1,13):
    files.append('chip%i_frw_e' % (j))
(uE,iE) = parseLvm2UI(folder,files)


folder = '/media/alex/Seagate Expansion Drive/Эксперименты/БЭП/Чистохин 12шт/25С/'
files = []
for j in range(1,13):
    files.append('chip%02i_frw_d' % (j))
(uD,iD) = parseLvm2UI(folder,files)

files = []
for j in range(1,13):
    files.append('chip%02i_frw_e' % (j))
(uE,iE) = parseLvm2UI(folder,files)


fig = plt.figure(1)
plt.clf()
fig.set_facecolor('white')
lgnd = []
Norm = (1,5,8)
for j in Norm: #9
    plt.plot(uD[j]*1e3,iD[j]*1e9,'-x')
    lgnd.append('%d D' % (j+1))
    plt.plot(uE[j]*1e3,iE[j]*1e9)
    lgnd.append('%d E' % (j+1))
plt.legend(lgnd, loc='best')
plt.xlabel('U, мВ')
plt.ylabel('I, нА')
plt.title('ВАХ')
plt.grid()
plt.show()


(Uoc,Isc) = findPrms(uE[1],iE[1])
Uoc*1e3
Isc*1e9





# сращиваем точки данных прямой и обратной ВАХ
IV_thick = [] # list

V = np.hstack((u[0], u[1]))
I = np.hstack((i[0], i[1]))
n = V.argsort()
V = V[n]
I = I[n]
plt.clf()
plt.plot(V*1e3,I*1e6)

IV_thick.append((V, I))

V = np.hstack((u[2], u[3]))
I = np.hstack((i[2], i[3]))
n = V.argsort()
V = V[n]
I = I[n]
IV_thick.append((V, I))

# Строим графики ВАХ всех чипов
fig = plt.figure(1)
plt.clf()
fig.set_facecolor('white')
lgnd = []
for j in range(2):
    plt.plot(IV_thick[0][0]*1e3,IV_thick[0][1]*1e6)
    lgnd.append('исх. #%d' % (j))
plt.legend(lgnd, loc='best')

# не показательно, делаем по отдельности - прямую и обратную
plt.clf()
lgnd = []
for j in range(2):
#    plt.plot(u0[j]*1e3,i0[j]*1e9)
#    lgnd.append('m%d исх.' % (j))
#    plt.plot(u1b[j]*1e3,i1b[j]*1e9)
#    lgnd.append('m%d утон.' % (j))
    plt.plot(u1[j]*1e3,i1[j]*1e9,'-x')
    lgnd.append('m%d утон.' % (j+1))
    plt.plot(uN[j]*1e3,iN[j]*1e9)
    lgnd.append('m%d утон.+Ni' % (j+1))


plt.xlim((0,50))
plt.ylim((0,5))
plt.legend(lgnd, loc='best')
plt.xlabel('U, мВ')
plt.ylabel('I, нА')
plt.title('ВАХ')
plt.grid()
plt.show()

(Uoc,Isc) = findPrms(uN,iN)
Uoc*1e3
Isc*1e9


plt.clf()
lgnd = []
pN = []
for j in range(2):
    pN.append(-iN[j]*uN[j])
    plt.plot(uN[j]*1e3,pN[j]*1e9)
    lgnd.append('m%d утон.+Ni' % (j+1))
plt.xlim((0,50))
plt.ylim((0,0.2))

pN[0].max()*1e9
pN[1].max()*1e9


print('max Uxx=%.2f мВ, на чипе %d' % (Uoc.max()*1e3, Uoc.argmax()+1 ))
print('max Iкз=%.2f нА, на чипе %d' % (Isc.min()*1e9, Isc.argmin()+1 ))

# Строим графики ВАХ всех чипов
fig = plt.figure(1)
plt.clf()
fig.set_facecolor('white')
lgnd = []
for j in range(len(u)):
    plt.plot(u[j]*1e3,i[j]*1e6)
    lgnd.append(files[j])
    # plt.plot(u[j]*1e3,-i[j]*1e9)

#plt.xlim((0,maxU*1.05*1e3))
plt.ylim((0,-minI*1.05*1e9))
plt.legend(lgnd, loc='best')
plt.xlabel('U, мВ')
plt.ylabel('I, мкА')
plt.title('ВАХ')
plt.grid()
plt.show()


# Строим графики ВАХ всех нормальных чипов - у которых Uxx>0, Isc<0
maxU = Uoc.max()
minI = Isc.min()

fig = plt.figure(1)
plt.clf()
fig.set_facecolor('white')
U = np.linspace(0,maxU,100)
lgnd = []
for j in range(len(u)):
    if Uoc[j]>0 and Isc[j]<0:
        fi = interp1d(u[j], i[j])
        plt.plot(U*1e3,-fi(U)*1e9)
        lgnd.append('%s -%d' % (files[j], j))
        # plt.plot(u[j]*1e3,-i[j]*1e9)

plt.xlim((0,maxU*1.05*1e3))
plt.ylim((0,-minI*1.05*1e9))
plt.legend(lgnd, loc='best')
plt.xlabel('U, мВ')
plt.ylabel('I, нА')
plt.title('ВАХ')
plt.grid()
plt.show()


#графики мощностей
fig = plt.figure(1)
plt.clf()
fig.set_facecolor('white')
lgnd = []
U = np.linspace(0,maxU,100)
maxP = np.array([])
FF = np.array([])
Iopt = np.array([])
Uopt = np.array([])
for j in range(len(u)):
    if Uoc[j]>0 and Isc[j]<0:
        fi = interp1d(u[j], i[j])
        P = -fi(U)*U
        plt.plot(U*1e3,P*1e9)
        #plt.plot(u[j]*1e3,-i[j]*1e9*u[j])
        lgnd.append(files[j])
        FF = np.append(FF, P.max()/np.abs(Uoc[j]*Isc[j]))
        k = P.argmax()
        Uopt = np.append(Uopt,U[k])
        Iopt = np.append(Iopt,-fi(U[k]))
        maxP = np.append(maxP,P.max())

plt.xlim((0,maxU*1.05e3))
plt.ylim((0,-minI*maxU*0.5e9*1))
plt.legend(lgnd, loc='best')
plt.xlabel('U, мВ')
plt.ylabel('P, нВт')
plt.title('Мощность')
plt.grid()
plt.show()
j = maxP.argmax()
print('max P=%.5f нВт (Uopt=%.3f mV, Iopt=%.3f nA), FF=%.2f, на чипе %s' % (maxP[j]*1e9, Uopt[j]*1e3, Iopt[j]*1e9, FF[j]*100, lgnd[j] ))

#ax = fig.add_subplot(111)
#plt.plot(u,i)
#plt.xlim((0,0.06))
#plt.grid(True)
#plt.draw_if_interactive()
#plt.show()
#ax.xaxis.set_view_interval([0, 0.06])
#ax.grid(True)

# Экспоненциальная аппроксимация
#import scipy as sp
from scipy.optimize.minpack import curve_fit
expFun = lambda u, a, b, c, d: a*np.exp(b*u)+c+d*u

def getExpFit(U,I):
    guess = [1e-9, 37, -1e-9, 1e6] #
    params, cov = curve_fit(expFun, U, I, p0 = guess)
    return params

j=5

for j in range(0,len(u)):
    U = u[j]
    I = i[j]

    a, b, c, d = getExpFit(U,I)
Uv = np.linspace(U.min(),U.max(),100)
Iv = expFun(Uv,a,b,c,d)
plt.clf()
plt.plot(U, I, 'b*')
plt.plot(Uv, Iv, 'r-')
plt.show()

#модель при кратном увеличении Ig (1мкм)
Ig18 = a-c
Ig82 = 82/18*Ig18
c82 = a - Ig82
Uvx = np.linspace(0,0.12,300)
Iv82 = expFun(Uvx,a,b,c82,d)
Iv18 = expFun(Uvx,a,b,c,d)
fi = interp1d(Iv82, Uvx)
Uoc82= float(fi(0))
Isc82 = Iv82[0]
P82 = -Uv82*Iv82
k = P82.argmax()
print('82: max P=%.5f нВт (Uopt=%.3f mV, Iopt=%.3f nA), FF=%.2f' % (P82[k]*1e9, Uv82[k]*1e3, Iv82[k]*1e9, P82[k]/(Uoc82*Isc82)*100 ))


plt.clf()
plt.plot(Uv82, Iv82, 'b-')
plt.plot(Uv, Iv, 'r-')
plt.show()

plt.clf()
plt.plot(Uvx*1e3, Iv82*1e9, 'r-')
plt.plot(Uvx*1e3, Iv18*1e9, 'b-')
plt.xlim((0,120))
plt.ylim((0,-40))
plt.xlabel('U, мВ')
plt.ylabel('I, нВт')
plt.title('ВАХ')
plt.show()


# Строим траекторию Точки МАкс Мощности
IgK = []
PmaxK = []
UoptK = []
IoptK = []
for K in range(82,15,-2):
    IgK = np.append(IgK,K/18*Ig18)
    cK = a - IgK[-1]
    IvK = expFun(Uvx,a,b,cK,d)
    PK = -Uvx*IvK
    k = PK.argmax()
    PmaxK = np.append(PmaxK, PK[k])
    UoptK = np.append(UoptK, Uvx[k])
    IoptK = np.append(IoptK, IvK[k])

kxFun = lambda u,R: -u/R
R, cov = curve_fit(kxFun, UoptK, IoptK, p0 = 70e-3/30e-9)
R = float(R[0])

plt.clf()
plt.plot(UoptK*1e3, IoptK*1e9, 'b-')
plt.plot(UoptK*1e3, -UoptK/R*1e9, 'r-')
plt.xlabel('U, мВ')
plt.ylabel('I, нВт')
# plt.title('')
plt.show()




# Модель Rs&Rp - не получилось
from scipy.optimize import minimize
from scipy.optimize import rosen
from math import exp

const_k = 1.3806226e-23
const_e = 1.6021918e-19
Vth = lambda T: const_k*T/const_e
Ivd = lambda Vd,n,T,I0: I0*(np.exp(Vd/(n*Vth(T)))-1)

def funDiodeMdlSpice(x, U, I, T):
    n  = x[0]
    I0 = x[1]
    Ig = x[2]
    Gs = x[3]
    Gp = x[4]

    Vd = -I/Gs + U
    nev = (U*Gs+Ig-Ivd(Vd,n,T,I0))/(Gp+Gs) - Vd
    return np.sum(nev**2)

x = [2, 3.6e-9, 1.8e-8, 1e8, 1.7e-8]
T=25+273.15
# Bounds for variables (only for L-BFGS-B, TNC and SLSQP).
res = minimize(funDiodeMdlSpice, x0, args=(U, I, T), method='SLSQP',
               bounds=((1, 5), (1e-10, 1e-6),(0, 1e-6), (0,None), (0,None)),
            tol=1e-25)

n  = res.x[0]
I0 = res.x[1]
Ig = res.x[2]
Rs = 1/res.x[3]
Rp = 1/res.x[4]
diodeMdlSpice(res.x, U, I, T)

def diodeMdlSpice(x, U, T):
    n  = x[0]
    I0 = x[1]
    Ig = x[2]
    Gs = x[3]
    Gp = x[4]
    Ux = []
    for j in range(len(U)):
        nev = lambda Vd: np.abs((U[j]*Gs+Ig-Ivd(Vd,n,T,I0))/(Gp+Gs) - Vd)
        res = minimize(nev, U[j], tol=1e-15)
        Ux.append(res.x[0])
    # возвращает ток
    return (U-Ux)*Gs

x = res.x
Im = diodeMdlSpice(x, U, T)

plt.clf()
plt.plot(U, I, 'b*')
plt.plot(U, Im, 'r-')
plt.show()