#encoding: utf-8
    
from scipy.optimize import curve_fit
import numpy as np
import itertools

def score(y,ya): #r**2
    return np.corrcoef(y,ya)[0, 1]**2

def poly(x, A, P):
    """Значення полінома з коефіцієнтами A і степнями P"""
    comp=[a*x**p for a,p in zip(A,P)] # доданки полінома
    return sum(comp) # значення полінома a[0]+a[1]*x+a[2]*x**2

def pfit(x, y, P, Z):
    """Апроксимація поліномом з степенями P з врахуванням вектора занулення Z"""
    def polyz(x, *A): # Поліном з коефіцієнтами A з їх зануленням вектором Z"""
        A=[a*z for a,z in zip(A,Z)] # коефіцієнти полінома з врахуванням занулення
        return poly(x, A, P)
        
    A, cov = curve_fit(polyz, x, y, p0=Z) # апроксимувати нелінійним методом найменших квадратів
    s=score(y,poly(x, A, P)) # внутрішній критерій - наскільки добре p описує точки x,y
    return A, s # коефіцієнти полінома і R**2

def crossValidation(x, y, P, Z):
    """Повертає узагальнений критерій"""
    # ділимо дані на групи
    A,s0=pfit(x,y,P,Z) # підгонка усього
    x1,y1=x[::-2],y[::-2] # непарні
    x2,y2=x[::2],y[::2] # парні
    A1,s01=pfit(x1,y1,P,Z) # підгонка групи 1
    s1=score(y2, poly(x2,A1,P))
    A2,s02=pfit(x2,y2,P,Z) # підгонка групи 2
    s2=score(y1, poly(x1,A2,P))
    s3=score(poly(x,A1,P), poly(x,A2,P))
    #return np.mean([s0,s3]) # або 
    return np.mean([s0,s1,s2,s3])
    #return s1
    #return np.mean([s01,s1]) # мабуть найпростіший випадок

def combi(x,y,Np=4):
    """Комбінаторний алгоритм індуктивної самоорганізації моделі
    x,y - координати емпіричних точок
    Np - максимальна степінь полінома"""
    P=range(Np+1) # степені полінома [0,1,2,3,...]
    res=[] # результати
    # усі можливі комбінації
    for Z in itertools.product([0,1],repeat=Np+1):
        #Z - список для занулення коефіцієнтів (1,1,0,...)
        if any(Z[1:]): # окрім поліномів f=0 та f=const
            res.append([Z, crossValidation(x,y,P,Z)]) # узагальнений критерій
    res.sort(key=lambda x: x[1], reverse=True) # сортуємо за спаданням score
    return res # відсортований список Z, score

def plot(x,y,Z,xmin=None,xmax=None):
    """Рисує дані і поліном Z"""
    Np=len(Z)-1
    P=range(Np+1)
    A,s=pfit(x, y, P, Z)
    print "A, R**2 = ", A,s
    import matplotlib.pyplot as plt
    plt.plot(x,y,'ro')
    if xmin==None: xmin=x.min()
    if xmax==None: xmax=x.max()
    x_=np.linspace(xmin,xmax,100)
    y_=poly(x_,A,P)
    plt.plot(x_,y_,'k-',linewidth=2)
    
    from scipy.integrate import cumtrapz, quad
    y_int = cumtrapz(y_, x_, initial=0) # значення первісної шляхом інтегрування
    print 'Площа під кривою=', quad(poly,xmin,xmax,args=(A,P))[0]
    plt.plot(x_,y_int,'k--',linewidth=2)
    plt.show()
    #print "integral, score=", score2(A,P)

##
# def score2(A,P):
#     """Додатковий критерій. Використовується для підгонок функцій густини розподілу"""
#     from scipy.integrate import quad
#     s, err = quad(poly, 0, 1, args=(A,P)) # інтегрувати в межах
#     # s повинна бути рівна 1
#     return s, np.exp(-(s-1)**2) # повертає від 0 до 1
#     #scipy.special.erf(x) #?

##
if __name__=='__main__': # приклад використання
    N=16 # кількість точок
    # координати емпіричних точок
    x = np.linspace(0, 10, N)
    y = np.array([0,2,1,4,5,7,6,7,8,7,9,9,8,9,9,9])
    Np=4 # степінь полінома
    res=combi(x,y,Np) # список моделей
    for i in res:
        print i
    plot(x,y,Z=res[0][0]) # рисуємо найкращий поліном