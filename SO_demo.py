import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


T=100
c1=0.5;c2=0.5
q=0.5
food=1;war=2;mating=3

#画图基本参数的设置
fig = plt.figure(figsize=(10, 5))
plt.rcParams["font.family"]="FangSong"
plt.title("蛇优化算法的搜索demo")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.ion()


#适应度计算
def fitness(a):
    a[1,:]=a[0,:]*np.sin(a[0,:]*10)+np.cos(a[0,:]*2)*a[0,:]+10
    return a

#根据适应度站队
def line(b):
    b=b.T[np.lexsort(-b)].T
    return b

#变换
def conversion(k,man,woman):
    man=line(man)
    woman=line(woman)
    if k==1:    #寻找食物的变化
        kid1 = man[0, 0] + c2 * np.exp(-man[1, 0] / man[1, :]) * rand.uniform(-5, 5, (2, 15))
        kid2 = woman[0, 0] + c2 * np.exp(-woman[1, 0] / woman[1, :]) * rand.uniform(-5, 5, (2, 15))
    if k==2:   #战斗的变化
        kid1 = man[0,:]+c2*np.exp(-man[1,0]/man[1,:])*rand.uniform(-1,1,(2,15))*(man[0, 0]-man[0,:])
        kid2 = woman[0,:]+c2*np.exp(-woman[1,0]/woman[1,:])*rand.uniform(-1,1,(2,15))*(woman[0,0]-woman[0,:])
    if k==3:    #交配的变化
        kid1 = man[0,:]+c2*np.exp(-woman[1,0]/man[1,:]) * rand.uniform(-1,1,(2,15))*(q*woman[0,:]-man[0,:])
        kid2 = woman[0,:]+c2*np.exp(-man[1,0]/woman[1,:])*rand.uniform(-1,1,(2,15))*(q*man[0,:]-woman[0,:])
    kid1=np.clip(kid1,0,5,out=None)
    kid2=np.clip(kid2,0,5,out=None)
    kid1=fitness(kid1)
    kid2=fitness(kid2)
    man=np.hstack((man,kid1))
    woman=np.hstack((woman,kid2))
    man=line(man)
    woman=line(woman)
    man=man[:,0:15]
    woman=woman[:,0:15]
    return man,woman

#设置主函数，主循环
def main():
    man = rand.uniform(0, 5, (2, 15))
    woman = rand.uniform(0, 5, (2, 15))
    man=fitness(man)
    woman=fitness(woman)
    for t in range(T):
        #画图的交互
        plt.cla()      #清空后画布
        s = np.arange(0, 5, 0.01)
        h = np.sin(10 * s) * s + np.cos(2 * s) * s + 10
        plt.plot(s, h, 'r--')
        if 'sca' in globals(): sca.remove()
        sca = plt.scatter(man[0,:], man[1,:], s=100, lw=0, c='red', alpha=0.5)
        plt.pause(0.1)      #停留时间

        print(t," 雄性蛇的适应度",man[1,0],"  雌性蛇的适应度",woman[1,0])
        #主循环
        temp=np.exp(-t/T)
        Q=c1*np.exp((t-T)/T)
        if Q<0.25:
            man,woman=conversion(food,man,woman)
            continue
        else:
            if temp>0.6:
                man,woman=conversion(food,man,woman)
                continue
            else:
                k=rand.rand()
                if k<0.6:
                    man,woman=conversion(war,man,woman)
                else:
                    man,woman=conversion(mating,man,woman)

    print(man[:,0])
    print(woman[:,0])
    plt.ioff()   #关闭交互模式
    plt.show()   #展示图片

if __name__=="__main__":
    main()