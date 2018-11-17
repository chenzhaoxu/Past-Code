#coding:utf-8
  
from numpy import *
import random
 
#-----------------------------------------------支持向量机----------------------------------------------------------

def svmloaddataset(filename):
    dataarr=[]
    labelarr=[]
    fr=open(filename)
    for line in fr.readlines():
        linearr=line.strip().split('\t')
        dataarr.append([float(linearr[0]),float(linearr[1])])
        labelarr.append(float(linearr[2]))
    return dataarr,labelarr

def selectrandj(i,m):
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipalpha(alpha,L,H):
    if alpha>H:   
        alpha=H
    if alpha<L:
        alpha=L
    return alpha

class osStruct(object):
    def __init__(self,xmat,ymat,C,tol):
        self.x=xmat
        self.y=ymat
        self.C=C
        self.tol=tol
        self.b=0.0
        self.m=shape(self.x)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.E=mat(zeros((self.m,2)))

def calcE(os,k):
    uk=float(multiply(os.y,os.alphas).T*(os.x*os.x[k].T))+os.b
    Ek=uk-os.y[k]
    return Ek

def updateEk(os,k):
    Ek=calcE(os,k)
    os.E[k]=[1,Ek]

def selectj(os,i,Ei):
    os.E[i]=[1,Ei]
    valid=nonzero(os.E[:,0].A)[0]
    j=-1
    maxdeltaE=0.0
    Ej=0
    if (len(valid))>1:
        for k in valid:
            if k==i:
                continue
            Ek=calcE(os,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxdeltaE:
                maxdeltaE=deltaE
                j=k
                Ej=Ek
        return j,Ej  
    else:
        j=selectrandj(i,os.m)
        Ej=calcE(os,j)
        return j,Ej

def initer(os,i):
    Ei=calcE(os,i)
    if ((os.y[i]*Ei<-os.tol) and (os.alphas[i]<os.C)) or ((os.y[i]*Ei>os.tol) and (os.alphas[i]>0)):
        j,Ej=selectj(os,i,Ei)
        alpha_i_old=os.alphas[i].copy()
        alpha_j_old=os.alphas[j].copy()
        if os.y[i]==os.y[j]:
            L=max(0,os.alphas[i]+os.alphas[j]-os.C)
            H=min(os.alphas[i]+os.alphas[j],os.C)
        else:
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        if L==H:
            print "L==H"
            return 0
        eta=os.x[i]*os.x[i].T+os.x[j]*os.x[j].T-2*os.x[i]*os.x[j].T
        if eta<=0.0:
            print "eta<=0.0"
            return 0
        os.alphas[j]+=os.y[j]*(Ei-Ej)/eta
        os.alphas[j]=clipalpha(os.alphas[j],L,H)
        updateEk(os,j)
        if abs(os.alphas[j]-alpha_j_old)<0.00001:
            print "j is not moving enough"
            return 0
        os.alphas[i]+=os.y[i]*os.y[j]*(alpha_j_old-os.alphas[j])
        updateEk(os,i)
        b1=os.b-Ei-os.y[i]*(os.alphas[i]-alpha_i_old)*os.x[i]*os.x[i].T-os.y[j]*(os.alphas[j]-alpha_j_old)*os.x[i]*os.x[j].T
        b2=os.b-Ej-os.y[i]*(os.alphas[i]-alpha_i_old)*os.x[i]*os.x[j].T-os.y[j]*(os.alphas[j]-alpha_j_old)*os.x[j]*os.x[j].T
        if os.alphas[i]>0 and os.alphas[i]<os.C:
            os.b=b1
        elif os.alphas[j]>0 and os.alphas[j]<os.C:
            os.b=b2
        else:
            os.b=(b1+b2)/2.0
        return 1
    else: 
        return 0

def outiter(dataarr,labelarr,C,tol,maxiter,kTup=('lin',0)):
    os=osStruct(mat(dataarr),mat(labelarr).T,C,tol)
    iter=0
    entireset=1
    alphaspairchanged=0
    while (iter<maxiter) and ((alphaspairchanged>0) or (entireset)):
        alphaspairchanged=0
        if entireset:
            for i in range(os.m):
                alphaspairchanged += initer(os,i)
                print "entireset: iter:%d, i:%d, alphaspairchanged:%d" % (iter,i,alphaspairchanged)
            iter+=1
        else:
            nobound=nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]
            for i in nobound:
                alphaspairchanged += initer(os,i)
                print "nobound: iter:%d, i:%d, alphaspairchanged:%d" % (iter,i,alphaspairchanged)
            iter+=1
        if entireset:
            entireset=0
        elif alphaspairchanged==0:
            entireset=1
        print "iternumber:%d" % iter
    return os.alphas,os.b

def calcw(x,y,alphas):
    x=mat(x) 
    y=mat(y).T
    m,n=shape(x)  
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*y[i],x[i].T)
    return w 
