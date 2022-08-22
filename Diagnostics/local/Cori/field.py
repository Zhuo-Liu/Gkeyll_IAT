import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm
from scipy.optimize import curve_fit

lz = 1.0
ly = 0.5

nz = 96
ny = 48

dz = lz/nz
dy = ly/ny

z_plot = np.linspace(0,lz,nz)
y_plot = np.linspace(0,ly,ny)
ZZ, YY = np.meshgrid(z_plot, y_plot, indexing= 'xy')
ZZ = np.transpose(ZZ)
YY = np.transpose(YY)

kz_plot  = 2.0*np.pi*np.linspace(-int(nz/2), int(nz/2-1), nz)/lz
ky_plot  = 2.0*np.pi*np.linspace(-int(ny/2), int(ny/2-1), ny)/ly
K_z, K_y = np.meshgrid(kz_plot, ky_plot, indexing = 'xy')
K_z = np.transpose(K_z)
K_y = np.transpose(K_y)

def k_dependence(Ez_k,Ey_k):
    E_k = np.zeros(49)
    for i in range(nz):
        for j in range(ny):
            k_square = (i-int(nz/2))**2 + (j-int(ny/2))**2
            k_star = np.sqrt(k_square)
            if k_star <= 48:
                decimal = k_star - int(k_star)
                #app_inx = int(decimal * N)
                Ek_square = Ez_k[i,j]**2 + Ey_k[i,j]**2
                if decimal >= 0.5:
                    E_k[int(k_star)+1] += Ek_square
                else:
                    E_k[int(k_star)] += Ek_square
    
    return E_k

def theta_dependence(Ez,Ey,N=100):
    E_theta = np.zeros(N+1)
    for z in range(nz):
        for y in range(ny):
            ez = Ez[z,y]
            ey = Ey[z,y]
            theta = np.abs(np.arctan(ey/ez))
            num  = int(theta/(np.pi/2/N))
            E_theta[num] = E_theta[num] + ez**2 + ey**2


    return E_theta

def new_theta_dependence(Ezk,Eyk,N=100):
    E_theta = np.zeros(101)
    for i in range(nz):
        for j in range(ny):
            kz = i-int(nz/2)
            ky = j-int(ny/2)
            k_square = kz**2 + ky**2
            k_star = int(np.sqrt(k_square))
            if k_star < 48:
                if Ezk[kz,int(ny/2)] > 0.1 and Eyk[int(nz/2),ky] > 0.1:
                    Ek_square = Ezk[kz,int(ny/2)]**2 + Eyk[int(nz/2),ky]**2
                else:
                    Ek_square = 0

                if kz ==0:
                    theta = np.pi/2
                else:
                    theta = np.abs(np.arctan(ky/kz))

                theta_num = int(theta/(np.pi/2/100))

                E_theta[theta_num] += Ek_square 
    
    return E_theta

def load_phi():
    Ez_k_list=[]
    Ey_k_list=[]
    Ez_list = []
    Ey_list = []
    for i in range(25,481):
        fignum = str(i).zfill(4)
        filename = './Diagnostics/local/Cori/mass25/rescheck/4/field/M25_E2_3_field_' + fignum + '.txt'
        phi = np.loadtxt(filename)
        E_z, E_y = np.gradient(phi)
        E_z = E_z/dz
        E_y = E_y/dy
        Ez_k = np.abs(np.fft.fftshift(np.fft.fftn(E_z)))
        Ey_k = np.abs(np.fft.fftshift(np.fft.fftn(E_y)))
        Ez_k_list.append(Ez_k)
        Ey_k_list.append(Ey_k)
        Ez_list.append(E_z)
        Ey_list.append(E_y)
    
    return np.array(Ez_k_list), np.array(Ey_k_list), np.array(Ez_list), np.array(Ey_list)

def phi_plot():
    phi = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/field/M25_E2_3_field_0480.txt')
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    ax.contour3D(ZZ, YY, phi, 100)

    p = ax.plot_surface(ZZ, YY, phi, rstride=4, cstride=4, linewidth=0,cmap=cm.coolwarm)

    # surface_plot with color grading and color bar
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    # cb = fig.colorbar(p, shrink=0.5)

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    cset = ax.contourf(ZZ, YY, phi, 30,  zdir='z', offset= -0.0004, cmap=matplotlib.cm.coolwarm)
    ax.set_zlim3d(-0.0004, 0.0006)
    plt.show()

def k_omega():
    _,_,Ez,Ey = load_phi()
    Ez1 = np.transpose(Ez[:50,:,24])
    Ez1fft = np.fft.fftshift(np.fft.rfft2(Ez1))

    omegas = np.fft.rfftfreq(50,d=(250)/(50-1))
    ks = np.linspace(-48,47,96)

    zz,yy = np.meshgrid(ks,omegas,indexing='xy')
    zz = np.transpose(zz)
    yy = np.transpose(yy)

    plt.contourf(zz,yy,np.abs(Ez1fft))
    plt.colorbar()
    plt.show()

    # ez1d = Ez[80,:,24]
    # ez1dfft = np.fft.fft(ez1d)
    # plt.plot(np.linspace(-48,47,96),np.abs(np.fft.fftshift(ez1dfft)))
    # plt.show()

def fit(Ek,x):

    def inline_function(x,a,b, c,d,e,f,g):
        return  a*x**6 + b*x**5 +c*x**4 +d*x**3 + e*x**2 + f*x +g

    popt, pcov = curve_fit(inline_function, x, Ek)

    newek = inline_function(x,*popt)

    return newek
    

def k_main():
    Ez_k_list, Ey_k_list,_,_ = load_phi()
    Ez_k_400 = Ez_k_list[80-25,:,:]
    Ey_k_400 = Ey_k_list[80-25,:,:]
    Ez_k_500 = Ez_k_list[100-25,:,:]
    Ey_k_500 = Ey_k_list[100-25,:,:]
    Ez_k_750 = Ez_k_list[160-25,:,:]
    Ey_k_750 = Ey_k_list[160-25,:,:]
    Ez_k_1200 = Ez_k_list[240-25,:,:]
    Ey_k_1200 = Ey_k_list[240-25,:,:]
    Ez_k_1800 = Ez_k_list[360-25,:,:]
    Ey_k_1800 = Ey_k_list[360-25,:,:]
    Ez_k_2400 = Ez_k_list[480-25,:,:]
    Ey_k_2400 = Ey_k_list[480-25,:,:]
    Ek_400 = k_dependence(Ez_k_400,Ey_k_400)
    Ek_500 = k_dependence(Ez_k_500,Ey_k_500)
    Ek_750 = k_dependence(Ez_k_750,Ey_k_750)
    Ek_1200 = k_dependence(Ez_k_1200,Ey_k_1200)
    Ek_1800 = k_dependence(Ez_k_1800,Ey_k_1800)
    Ek_2400 = k_dependence(Ez_k_2400,Ey_k_2400)
    # k_plot, Ek_750_f = fit(Ek_750)

    k_list = np.arange(0.6,48,0.1)*np.pi*2
    N_k = 1/(k_list*k_list*k_list) * np.log(1/k_list/0.02) 
    kf = np.arange(0.0,48,0.01)*np.pi*2
    krde = kf*0.02
    N_k_full = 1/(kf*kf*kf)*(1+krde*krde)**(-3/2)*(np.log(np.sqrt(1+krde*krde)/krde)-0.5/(1+krde*krde)-0.25/(1+krde*krde)/(1+krde*krde))

    k_plot_2 = np.arange(49)

    plt.figure(figsize=(8,6))
    plt.plot(k_plot_2,Ek_400,label=r'$\omega_{pe}t=400$',linewidth=3)
    #plt.plot(k_plot_2,Ek_500,label=r'\omega_{pe}t=500$')
    plt.plot(k_plot_2,Ek_750,label=r'$\omega_{pe}t=750$',linewidth=3)
    #plt.plot(k_plot_2,Ek_1200,label=r'\omega_{pe}t=1200$')
    plt.plot(k_plot_2,Ek_1800,label=r'$\omega_{pe}t=1600$',linewidth=3)
    plt.plot(k_plot_2,Ek_2400,label=r'$\omega_{pe}t=2400$',linewidth=3)
    plt.plot(k_list,N_k*700,label='theory',linewidth=3,linestyle = '--')
    #plt.plot(k_list,N_k*200,label='theory2')
    #plt.plot(kf,N_k_full*900,label='theory2')
    plt.legend(fontsize=16)
    plt.xlim(0,20)
    plt.ylim(1e-4,100)
    plt.xlabel(r'$kd_e / 2\pi $',fontsize=20)
    plt.ylabel(r'$N(k)$',fontsize=20)
    plt.tick_params(labelsize=14)
    plt.yscale('log')
    plt.show()



def theta_main():
    def phi(x):
        return (4*x**2 - 3*x**3) / (1-x + 0.003)**2
    
    theta = np.arange(0.1,np.pi/2,0.01)
    cos_theta = np.cos(theta)
    N_theta = np.array([phi(cos) for cos in cos_theta])

    Ez_k_list, Ey_k_list, Ez_list, Ey_list = load_phi()
    Ez_400 = Ez_list[80-25,:,:]
    Ey_400 = Ey_list[80-25,:,:]
    Ez_500 = Ez_list[120-25,:,:]
    Ey_500 = Ey_list[120-25,:,:]
    Ez_800 = Ez_list[160-25,:,:]
    Ey_800 = Ey_list[160-25,:,:]
    Ez_1200 = Ez_list[180-25,:,:]
    Ey_1200 = Ey_list[180-25,:,:]
    Ez_1600 = Ez_list[360-25,:,:]
    Ey_1600 = Ey_list[360-25,:,:]
    Ez_2400 = Ez_list[480-25,:,:]
    Ey_2400 = Ey_list[480-25,:,:]
    Ezk_750 = Ez_k_list[160-25,:,:]
    Eyk_750 = Ey_k_list[160-25,:,:]

    E_400 = theta_dependence(Ez_400,Ey_400,50)
    E_500 = theta_dependence(Ez_500,Ey_500,50)
    E_800 = theta_dependence(Ez_800,Ey_800,50)
    E_1200 = theta_dependence(Ez_1200,Ey_1200,50)
    E_1600 = theta_dependence(Ez_1600,Ey_1600,50)
    E_2400 = theta_dependence(Ez_2400,Ey_2400,50)
    E_400_f = fit(E_400,np.arange(51)*90/50)
    E_500_f = fit(E_500,np.arange(51)*90/50)
    E_800_f = fit(E_800,np.arange(51)*90/50)
    E_1200_f = fit(E_1200,np.arange(51)*90/50)
    E_1600_f = fit(E_1600,np.arange(51)*90/50)
    E_2400_f = fit(E_2400,np.arange(51)*90/50)

    #E_750_new = new_theta_dependence(Ezk_750,Eyk_750)


    theta_plot = np.arange(51)*90/50

    plt.figure(figsize=(8,6))
    #plt.plot(theta/np.pi*2*90, N_theta/1e4,label='theory')
    plt.plot(theta_plot,E_400_f/E_400_f[0],label=r'$\omega_{pe}t=400$')
    plt.plot(theta_plot,E_500_f/E_500_f[0],label=r'$\omega_{pe}t=600$')
    #plt.plot(k_plot_2,Ek_500,label=r'$\omega_{pe}t=500$')
    #plt.plot(theta_plot,E_750,label=r'$\omega_{pe}t=750$')
    plt.plot(theta_plot,E_800_f/E_800_f[0],label=r'$\omega_{pe}t=800$')
    #plt.plot(theta_plot,E_750_new/E_750_new[0],label=r'$\omega_{pe}t=750 new$')
    plt.plot(theta_plot,E_1200_f/E_1200_f[0],label=r'$\omega_{pe}t=900$')
    plt.plot(theta_plot,E_1600_f/E_1600_f[0],label=r'$\omega_{pe}t=1800$')
    #plt.plot(theta_plot,E_2400_f/E_2400_f[0],label=r'$\omega_{pe}t=2400$')
    # plt.plot(theta_list,N_k*700,label='theory')
    # plt.plot(theta_list,N_k*200,label='theory2')
    #plt.plot(kf,N_k_full*900,label='theory2')
    plt.legend(fontsize=16)
    #plt.xlim(0,20)
    #plt.ylim(1e-4,100)
    plt.xlabel(r'$\theta ^\circ$',fontsize=20)
    plt.ylabel(r'$N(\theta) / N(0)$',fontsize=20)
    plt.tick_params(labelsize=14)
    plt.show()

def test():
    kz  = 2.0*np.pi*np.linspace(-int(48/2), int(48/2-1), 48)
    ky  = 2.0*np.pi*np.linspace(-int(48/2), int(48/2-1), 48)
    KZ, KY = np.meshgrid(kz, ky, indexing = 'xy')
    KZ = np.transpose(KZ)
    KY = np.transpose(KY)

    myEz = np.zeros((48,48))
    myEy = np.zeros((48,48))
    phi = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            myEz[i,j] = np.cos(2*np.pi*3*i/48.0)
            myEy[i,j] = np.cos(2*np.pi*3*j/48.0)
            phi[i,j] = np.sin(2*np.pi*2*i/48) + np.sin(2*np.pi*j/48)
    
    myEzk = np.abs(np.fft.fftshift(np.fft.fftn(myEz)))
    myEyk = np.abs(np.fft.fftshift(np.fft.fftn(myEy)))
    #plt.contourf(KZ/2/np.pi,KY/2/np.pi,myEzk)
    # plt.contourf(KZ/2/np.pi,KY/2/np.pi,myEyk)
    # #plt.contourf(KZ/2/np.pi,KY/2/np.pi,phi)
    # #phik = np.fft.fftshift(np.fft.fft2(phi))
    # #plt.contourf(KZ/2/np.pi,KY/2/np.pi,phik)

    # plt.show()

    E_theta = np.zeros((101))
    for i in range(48):
        for j in range(48):
            kz = i-int(48/2)
            ky = j-int(48/2)
            k_square = kz**2 + ky**2
            k_star = int(np.sqrt(k_square))
            if k_star < 48:
                if myEzk[kz,24] > 1 and myEyk[24,ky] > 1:
                    Ek_square = myEzk[kz,24]**2 + myEyk[24,ky]**2
                else:
                    Ek_square = 0

                if kz ==0:
                    theta = np.pi/2
                else:
                    theta = np.abs(np.arctan(ky/kz))

                theta_num = int(theta/(np.pi/2/100))

                E_theta[theta_num] += Ek_square 


    plt.plot(np.arange(101)/100*90,E_theta)
    plt.show()


if __name__ == '__main__':
    #main()
    phi_plot()
    #k_main()
    #theta_main()
    #test()
    #k_omega()