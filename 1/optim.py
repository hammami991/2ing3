import numpy as np
import matplotlib.pyplot as plt
#utilisez intervalle  dans R2 [ax,bx][ay,by]
def GrapheLN(ax,bx,ay,by,N):
    # Si vous voulez utiliser sin,cos,exp etc ... utlisez numpy exemple : np.sin(x+y)
    f=eval('lambda x,y: ' + input("Entrer fonction:"))
    X, Y = np.meshgrid(np.linspace(ax,bx,N), np.linspace(ay,by,N))
    Z = f(X,Y)
    plt.figure(1)
    plt.title("Tracé approché du graphe")
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,Y,Z)
    plt.figure(2)
    graphe = plt.contour(X,Y,Z,N)
    plt.clabel(graphe,inline=1,fontsize=5,fmt='%3.2f')
    plt.title("Tracé approché des lignes de niveaux")
    plt.grid()
    plt.show()
GrapheLN(-1,1,-1,2,20)