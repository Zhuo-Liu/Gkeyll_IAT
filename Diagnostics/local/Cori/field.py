import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

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


def real_theta_dependence(Ez,Ey,N=100):

    E_theta = np.zeros(N+1)
    for z in range(nz):
        for y in range(ny):
            ez = Ez[z,y]
            ey = Ey[z,y]
            theta = np.abs(np.arctan(ey/ez))
            num  = int(theta/(np.pi/2/N))
            E_theta[num] = E_theta[num] + ez**2 + ey**2


    return E_theta

def load_phi():
    Ez_k_list=[]
    Ey_k_list=[]
    Ez_list = []
    Ey_list = []
    phi_list = []
    for i in range(200,3798):
        fignum = str(i).zfill(4)
        filename = './Cori/mass25/rescheck/1/field/M25_E2_0_field_' + fignum + '.txt'
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
        phi_list.append(phi)

    # for i in range(25,493):
    #     fignum = str(i).zfill(4)
    #     filename = './Cori/mass25/rescheck/4/field/M25_E2_3_field_' + fignum + '.txt'
    #     phi = np.loadtxt(filename)
    #     E_z, E_y = np.gradient(phi)
    #     E_z = E_z/dz
    #     E_y = E_y/dy
    #     Ez_k = np.abs(np.fft.fftshift(np.fft.fftn(E_z)))
    #     Ey_k = np.abs(np.fft.fftshift(np.fft.fftn(E_y)))
    #     Ez_k_list.append(Ez_k)
    #     Ey_k_list.append(Ey_k)
    #     Ez_list.append(E_z)
    #     Ey_list.append(E_y)
    #     phi_list.append(phi)

    # for i in range(25,493):
    #     fignum = str(i).zfill(4)
    #     filename = './Diagnostics/local/massRatio/25/field/M25_E2_3_field_' + fignum + '.txt'
    #     phi = np.loadtxt(filename)
    #     E_z, E_y = np.gradient(phi)
    #     E_z = E_z/dz
    #     E_y = E_y/dy
    #     Ez_k = np.abs(np.fft.fftshift(np.fft.fftn(E_z)))
    #     Ey_k = np.abs(np.fft.fftshift(np.fft.fftn(E_y)))
    #     Ez_k_list.append(Ez_k)
    #     Ey_k_list.append(Ey_k)
    #     Ez_list.append(E_z)
    #     Ey_list.append(E_y)
    #     phi_list.append(phi)

    
    return np.array(Ez_k_list), np.array(Ey_k_list), np.array(Ez_list), np.array(Ey_list), np.array(phi_list)

def phi_plot():
    phi = np.loadtxt('./Cori/mass25/rescheck/4/field/M25_E2_3_field_0200.txt')
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
    ax.set_zlim3d(-0.0004, 0.0003)
    plt.show()

def k_omega():
    _,_,Ez0, Ey, phi = load_phi()

    #Ez0 = np.transpose(Ez0[500:3500,:,12])
    #ts = np.arange(700,3700)/2
    #zs = np.linspace(0,1.0,48)
    #tt, zzt = np.meshgrid(ts, zs, indexing= 'xy')
    #tt = np.transpose(tt)
    #zzt = np.transpose(zzt)

    # plt.figure(figsize=(10,3))
    # plt.contourf(tt,zzt,Ez0)
    # plt.colorbar()
    # plt.savefig('./gg.jpg')
    # plt.show()

    #Ez1 = np.transpose(Ey[:800,:,12])
    #Ez1fft = np.fft.fftshift(np.fft.rfft2(Ez1))
    Ez = np.transpose(Ez0[3000:3700,:,0]) # 48 x 400
    Ezw = np.fft.rfft2(Ez) # 48 x 400
    Ezw2 = np.zeros_like(Ezw)
    for i in range(Ezw.shape[1]):
        Ezw2[:,i] = np.fft.fftshift(Ezw[:,i])
    absEzw = np.power(np.absolute(Ezw2),2)[:,:]  
    omegas = np.fft.rfftfreq(700,d=(350)/(700-1)) * 2.0 * np.pi
    ks = np.linspace(-24,23,48)

    k_list_25 = 2*np.pi*np.array([0, 0.3183, 1,      2,      4,     6,    8,     10,    12,     16,     18])
    gamma_list_25 =     np.array([0, 0.00143,0.00441,0.00833,0.0135,0.015,0.0141,0.0122,0.00973,0.00364,-0.0003])
    omega_list_25 =     np.array([0, 0.00835,0.0261, 0.051,  0.0946,0.128,0.153, 0.1718,0.187,  0.2122, 0.2237])
    f_25 = interp1d(k_list_25, gamma_list_25, kind='cubic')
    fw_25 = interp1d(k_list_25, omega_list_25, kind='cubic')
    k_sample_25 = np.arange(0.1,113,0.1)
    gamma_sample_25 = f_25(k_sample_25)
    omega_sample_25 = fw_25(k_sample_25)

    zz,yy = np.meshgrid(ks,omegas[:100],indexing='xy')
    zz = np.transpose(zz)
    yy = np.transpose(yy)
    absEzw = np.log(absEzw)
    absEzw[absEzw<-3] = -3
    #levels = [1e-9,1e-6,1e-3,1e-2,1e-1,1,3]

    fig     = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.15, 0.15, 0.80, 0.76])
    ax.set_ylim(0,1.0)
    #plt.clim(vmin=-10,vmax=30)
    levels = np.linspace(-3,3,21)
    pos = ax.contourf(zz,yy,absEzw[:,:100],levels=levels)
    #ax.plot(-k_sample_25/2/np.pi, omega_sample_25,linewidth=5, color='black',linestyle='--')
    ax.plot(-k_sample_25/2/np.pi, k_sample_25/2/np.pi * 0.15,linewidth=5, color='black',linestyle='--')
    ax.set_xlabel(r'$kd_e/2\pi$',fontsize=32)
    ax.set_ylabel(r'$\omega/\omega_{pe}$',fontsize=32)
    ax.text(-2,0.7,r'$\frac{\omega}{k} = 1.2v_{Te}$',fontsize=36,color='white')
    #plt.show()
    ax.tick_params(labelsize = 26)
    fig.colorbar(pos,ax=ax)
    plt.savefig('./out.jpg')

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
    

def k_main(Ez_k_list,Ey_k_list):
    def k_dependence(Ez_k,Ey_k):
        E_k = np.zeros(nz//2+1)
        for i in range(nz):
            for j in range(ny):
                k_square = (i-int(nz/2))**2 + (j-int(ny/2))**2
                k_star = np.sqrt(k_square)
                if k_star <= nz//2:
                    decimal = k_star - int(k_star)
                    Ek_square = Ez_k[i,j]**2 + Ey_k[i,j]**2
                    if decimal >= 0.5:
                        E_k[int(k_star)+1] += Ek_square
                    else:
                        E_k[int(k_star)] += Ek_square
        
        return E_k

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
    plt.plot(k_plot_2,Ek_1200,label=r'$\omega_{pe}t=1200$',linewidth=3)
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


def theta_main(Ez_k_list,Ey_k_list, N=6):
    def phi(x):
        return (4*x**2 - 3*x**3) / (1-x + 0.003)**2
    
    def theta_dependence(Ezk,Eyk,N=6):
        E_theta = np.zeros((N+1))
        delta_theta = np.pi/2/N
        for i in range(nz):
            for j in range(ny):
                kz = i-int(nz/2)
                ky = j-int(ny/2)
                Ek_square = Ezk[i,j]**2 + Eyk[i,j]**2

                if kz ==0:
                    theta_num = N-1
                else:
                    theta = np.abs(np.arctan(ky/kz))
                    theta_num = int(theta/delta_theta)
                    decimal = (theta - delta_theta*theta_num) / delta_theta
                    if decimal >= 0.5:
                        theta_num = theta_num + 1

                E_theta[theta_num] += Ek_square 
        
        return E_theta

    theta = np.arange(0.1,np.pi/2,0.01)
    cos_theta = np.cos(theta)
    N_theta = np.array([phi(cos) for cos in cos_theta])
    Ezk_400 = Ez_k_list[80-25,:,:]
    Eyk_400 = Ey_k_list[80-25,:,:]
    Ezk_750 = Ez_k_list[150-25,:,:]
    Eyk_750 = Ey_k_list[150-25,:,:]
    Ezk_1200 = Ez_k_list[240-25,:,:]
    Eyk_1200 = Ey_k_list[240-25,:,:]
    Ezk_1600 = Ez_k_list[320-25,:,:]
    Eyk_1600 = Ey_k_list[320-25,:,:]
    Ezk_2400 = Ez_k_list[480-25,:,:]
    Eyk_2400 = Ey_k_list[480-25,:,:]

    E_400 = theta_dependence(Ezk_400,Eyk_400,N)
    E_750 = theta_dependence(Ezk_750,Eyk_750,N)
    E_1200 = theta_dependence(Ezk_1200,Eyk_1200,N)
    E_1600 = theta_dependence(Ezk_1600,Eyk_1600,N)
    E_2400 = theta_dependence(Ezk_2400,Eyk_2400,N)

    # E_400_f = fit(E_400,np.arange(51)*90/50)
    # E_500_f = fit(E_500,np.arange(51)*90/50)
    # E_800_f = fit(E_800,np.arange(51)*90/50)
    # E_1200_f = fit(E_1200,np.arange(51)*90/50)
    # E_1600_f = fit(E_1600,np.arange(51)*90/50)
    # E_2400_f = fit(E_2400,np.arange(51)*90/50)
    #E_750_new = new_theta_dependence(Ezk_750,Eyk_750)


    theta_plot = np.arange(N+1)*90/N

    plt.figure(figsize=(8,6))
    #plt.plot(theta/np.pi*2*90, N_theta/1e4,label='theory')
    #plt.plot(theta_plot,E_2400/E_2400[0],label=r'$\omega_{pe}t=400$')
    #plt.plot(theta_plot,E_500/E_500[0],label=r'$\omega_{pe}t=600$')
    #plt.plot(k_plot_2,Ek_500,label=r'$\omega_{pe}t=500$')
    plt.plot(theta_plot,E_400/E_400[0],label=r'$\omega_{pe}t=400$',linewidth=3)
    plt.plot(theta_plot,E_750/E_750[0],label=r'$\omega_{pe}t=750$',linewidth=3)
    plt.plot(theta_plot,E_1200/E_1200[0],label=r'$\omega_{pe}t=1200$',linewidth=3)
    plt.plot(theta_plot,E_1600/E_1600[0],label=r'$\omega_{pe}t=1600$',linewidth=3)
    plt.plot(theta_plot,E_2400/E_2400[0],label=r'$\omega_{pe}t=2400$',linewidth=3)
    # plt.plot(theta_list,N_k*700,label='theory')
    # plt.plot(theta_list,N_k*200,label='theory2')
    #plt.plot(kf,N_k_full*900,label='theory2')
    plt.legend(fontsize=16)
    #plt.xlim(0,20)
    #plt.ylim(1e-4,100)
    plt.xlabel(r'$\theta ^\circ$',fontsize=20)
    plt.ylabel(r'$N(\theta) / N(0)$',fontsize=20)
    plt.xlim(0,90)
    plt.tick_params(labelsize=14)
    plt.show()


def k_theta_main(Ez_k_list, Ey_k_list, N_t, N_k):
    def k_theta_dependence(Ezk,Eyk, N_theta=6, N_k = 48):
        W_k_theta = np.zeros((N_k+1,N_theta+1))

        delta_theta = np.pi/2/N_theta
        delta_k = 48/N_k

        for i in range(nz):
            for j in range(ny):
                kz = i-int(nz/2)
                ky = j-int(ny/2)
                k_square = kz**2 + ky**2
                k_star = np.sqrt(k_square)
                if k_star<48:
                    #Ek_square = Ezk[i,j]**2 + Eyk[i,j]**2
                    Ek_square = Ezk[i,j]**2
                    
                    k_num = int(k_star/delta_k)
                    decimal_k = (k_star - delta_k*k_num) / delta_k
                    if decimal_k >=0.5:
                        k_num = k_num+1

                    # if decimal_k >= 0.5:
                    #     k_num = int(k_star) + 1
                    # else:
                    #     k_num = int(k_star)

                    if kz == 0:
                        theta_num = N_theta
                    else:
                        theta = np.abs(np.arctan(ky/kz))
                        theta_num = int(theta/delta_theta)
                        decimal = (theta - delta_theta*theta_num) / delta_theta
                        if decimal >= 0.5:
                            theta_num = theta_num + 1
                    
                    W_k_theta[k_num, theta_num] += Ek_square

        return W_k_theta

    Ez_k_400 = Ez_k_list[60-25,:,:]
    Ey_k_400 = Ey_k_list[60-25,:,:]
    Ez_k_500 = Ez_k_list[100-25,:,:]
    Ey_k_500 = Ey_k_list[100-25,:,:]
    Ez_k_800 = Ez_k_list[160-25,:,:]
    Ey_k_800 = Ey_k_list[160-25,:,:]
    Ez_k_1200 = Ez_k_list[240-25,:,:]
    Ey_k_1200 = Ey_k_list[240-25,:,:]
    Ez_k_1800 = Ez_k_list[360-25,:,:]
    Ey_k_1800 = Ey_k_list[360-25,:,:]
    Ez_k_2400 = Ez_k_list[480-25,:,:]
    Ey_k_2400 = Ey_k_list[480-25,:,:]

    W_ktheta_400 = k_theta_dependence(Ez_k_400,Ey_k_400, N_t, N_k)

    ks = np.arange(N_k+1)/N_k*48
    ts = np.arange(N_t+1)/N_t*90

    kk, tt = np.meshgrid(ks,ts,indexing='xy')
    kk = np.transpose(kk)
    tt = np.transpose(tt)

    plt.contourf(kk[:N_k//4,],tt[:N_k//4,],W_ktheta_400[:N_k//4,])
    #plt.contourf(kk,tt,W_ktheta_400)
    plt.colorbar()
    plt.show()


def test():
    kz  = 2.0*np.pi*np.linspace(-int(96/2), int(96/2-1), 96)
    ky  = 2.0*np.pi*np.linspace(-int(48/2), int(48/2-1), 48)
    KZ, KY = np.meshgrid(kz, ky, indexing = 'xy')
    KZ = np.transpose(KZ)
    KY = np.transpose(KY)

    myEz = np.zeros((96,48))
    myEy = np.zeros((96,48))
    phi = np.zeros((96,48))
    for i in range(96):
        for j in range(48):
            myEz[i,j] = 2*np.cos(2*np.pi*(2*i/96.0+j/48.0)) + np.cos(2*np.pi*(3*i/96.0+3*j/48.0))
            myEy[i,j] = np.cos(2*np.pi*(2*i/96.0+j/48.0))
            #phi[i,j] = np.sin(2*np.pi*2*i/48) + np.sin(2*np.pi*j/48)
            phi[i,j] = np.sin(2*np.pi*(2*i/96.0+j/48.0))
    
    myEzk = np.abs(np.fft.fftshift(np.fft.fftn(myEz)))
    myEyk = np.abs(np.fft.fftshift(np.fft.fftn(myEy)))
    # plt.contourf(KZ/2/np.pi,KY/2/np.pi,myEzk)
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    # plt.show()
    # plt.clf()
    # plt.contourf(KZ/2/np.pi,KY/2/np.pi,myEyk)
    # plt.show()
    # plt.clf()
    #plt.contourf(KZ/2/np.pi,KY/2/np.pi,myEz)
    #phik = np.fft.fftshift(np.fft.fft2(phi))
    #plt.contourf(KZ/2/np.pi,KY/2/np.pi,phik)

    plt.show()

    E_theta = np.zeros((101))
    for i in range(96):
        for j in range(48):
            kz = i-int(96/2)
            ky = j-int(48/2)
            k_square = kz**2 + ky**2
            k_star = int(np.sqrt(k_square))
            Ek_square = myEzk[i,j]**2 + myEyk[i,j]**2

            if kz ==0:
                theta = np.pi/2
            else:
                theta = np.abs(np.arctan(ky/kz))

            theta_num = int(theta/(np.pi/2/100))

            E_theta[theta_num] += Ek_square 


    plt.plot(np.arange(101)/100*90,E_theta)
    plt.show()

def smooth(stock_col,WSZ):
    out0 = np.convolve(stock_col,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(stock_col[:WSZ-1])[::2]/r
    stop = (np.cumsum(stock_col[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start,out0,stop))

if __name__ == '__main__':
    phi_plot()
    #k_omega()
    #Ez_k_list, Ey_k_list,Ez_list,Ey_list,_ = load_phi()

    # W = 0
    # for i in range(96):
    #     for j in range(48):
    #         W = W + Ez_list[100,i,j] * Ez_list[100,i,j] + Ey_list[100,i,j] * Ey_list[100,i,j]
    # W = W / 96 / 48 * 0.5
    # print(W)


    #theta_main(Ez_k_list, Ey_k_list,6)
    #k_main(Ez_k_list,Ey_k_list)
    #k_theta_main(Ez_k_list,Ey_k_list,100,400)