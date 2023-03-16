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

def load_phi():
    Ez_k_list=[]
    Ey_k_list=[]
    Ez_list = []
    Ey_list = []
    phi_list = []
    for i in range(0,450):
        fignum = str(i).zfill(4)
        filename = './massRatio/mass100/E5_H2/field/M100_E5_field_' + fignum + '.txt'
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
    
    return np.array(Ez_k_list), np.array(Ey_k_list), np.array(Ez_list), np.array(Ey_list), np.array(phi_list)


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

    Ez_k_500 = Ez_k_list[50,:,:]
    Ey_k_500 = Ey_k_list[50,:,:]
    Ez_k_1000 = Ez_k_list[100,:,:]
    Ey_k_1000 = Ey_k_list[100,:,:]
    Ez_k_1800 = Ez_k_list[200,:,:]
    Ey_k_1800 = Ey_k_list[200,:,:]
    Ez_k_3500 = Ez_k_list[350,:,:]
    Ey_k_3500 = Ey_k_list[350,:,:]
    Ez_k_4300 = Ez_k_list[420,:,:]
    Ey_k_4300 = Ey_k_list[420,:,:]
    # Ez_k_2400 = Ez_k_list[480-25,:,:]
    # Ey_k_2400 = Ey_k_list[480-25,:,:]
    Ek_500 = k_dependence(Ez_k_500,Ey_k_500)
    Ek_1000 = k_dependence(Ez_k_1000,Ey_k_1000)
    Ek_1800 = k_dependence(Ez_k_1800,Ey_k_1800)
    Ek_3500 = k_dependence(Ez_k_3500,Ey_k_3500)
    Ek_4300 = k_dependence(Ez_k_4300,Ey_k_4300)
    # k_plot, Ek_750_f = fit(Ek_750)

    k_list = np.arange(0.6,48,0.1)*np.pi*2
    N_k = 1/(k_list*k_list*k_list) * np.log(1/k_list/0.02) 
    kf = np.arange(0.0,48,0.01)*np.pi*2
    krde = kf*0.02
    N_k_full = 1/(kf*kf*kf)*(1+krde*krde)**(-3/2)*(np.log(np.sqrt(1+krde*krde)/krde)-0.5/(1+krde*krde)-0.25/(1+krde*krde)/(1+krde*krde))

    k_plot_2 = np.arange(49)

    fig = plt.figure(figsize=(9,7))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
    ax.plot(k_plot_2,Ek_500,label=r'$\omega_{pe}t=500$',linewidth=3)
    ax.plot(k_plot_2,Ek_1000,label=r'$\omega_{pe}t=1000$',linewidth=3)
    ax.plot(k_plot_2,Ek_1800,label=r'$\omega_{pe}t=1800$',linewidth=3)
    ax.plot(k_plot_2,Ek_3500,label=r'$\omega_{pe}t=3500$',linewidth=3)
    ax.plot(k_plot_2,Ek_4300,label=r'$\omega_{pe}t=4300$',linewidth=3)
    # plt.plot(k_plot_2,Ek_2400,label=r'$\omega_{pe}t=2400$',linewidth=3)
    ax.plot(k_list,N_k*1700,linewidth=3,linestyle = '--',color='black')
    #plt.plot(k_list,N_k*700,label='theory2',linewidth=3,linestyle = '--',color='black')
    #plt.plot(k_list,N_k*200,label='theory2')
    #plt.plot(kf,N_k_full*900,label='theory2')
    ax.legend(fontsize=20)
    ax.set_xlim(0,20)
    ax.set_ylim(1e-4,1000)
    ax.set_xlabel(r'$kd_e / 2\pi $',fontsize=26)
    ax.set_ylabel(r'$N(k)$',fontsize=26)
    ax.tick_params(labelsize=22)
    ax.set_yscale('log')
    plt.show()


def theta_main(Ez_k_list,Ey_k_list, N=6):
    def phi(x):
        return (4*x**2 - 3*x**3) / (1-x + 0.003)**2

    def fit(Ek,x):

        def inline_function(x,a,b, c,d,e,f,g):
            #return  a*x**6 + b*x**5 +c*x**4 +d*x**3 + e*x**2 + f*x +g
            return c*x**4 +d*x**3 + e*x**2 + f*x +g
        popt, pcov = curve_fit(inline_function, x, Ek)

        newek = inline_function(x,*popt)

        return newek

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
    Ezk_500 = Ez_k_list[40,:,:]
    Eyk_500 = Ey_k_list[40,:,:]
    Ezk_1000 = Ez_k_list[100,:,:]
    Eyk_1000 = Ey_k_list[100,:,:]
    Ezk_3500 = Ez_k_list[340,:,:]
    Eyk_3500 = Ey_k_list[340,:,:]
    Ezk_4300 = Ez_k_list[430,:,:]
    Eyk_4300 = Ey_k_list[430,:,:]

    E_500 = theta_dependence(Ezk_500,Eyk_500,N)
    E_1000 = theta_dependence(Ezk_1000,Eyk_1000,N)
    E_3500 = theta_dependence(Ezk_3500,Eyk_3500,N)
    E_4300 = theta_dependence(Ezk_4300,Eyk_4300,N)
    #E_2400 = theta_dependence(Ezk_2400,Eyk_2400,N)

    E_500_f = fit(E_500,np.arange(51)*90/50)
    E_1000_f = fit(E_1000,np.arange(51)*90/50)
    E_3500_f = fit(E_3500,np.arange(51)*90/50)
    E_4300_f = fit(E_4300,np.arange(51)*90/50)


    theta_plot = np.arange(N+1)*90/N

    fig = plt.figure(figsize=(9,7))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
    # plt.plot(theta_plot,E_500/E_500[0],label=r'$\omega_{pe}t=500$',linewidth=3)x
    # plt.plot(theta_plot,E_1000/E_1000[0],label=r'$\omega_{pe}t=1000$',linewidth=3)
    # plt.plot(theta_plot,E_3500/E_3500[0],label=r'$\omega_{pe}t=3500$',linewidth=3)
    # plt.plot(theta_plot,E_4300/E_4300[0],label=r'$\omega_{pe}t=4300$',linewidth=3)
    ax.plot(theta_plot,E_500_f/E_500_f[0],label=r'$\omega_{pe}t=500$',linewidth=3)
    ax.plot(theta_plot,E_1000_f/E_1000_f[0],label=r'$\omega_{pe}t=1000$',linewidth=3)
    ax.plot(theta_plot,E_3500/E_3500[0],label=r'$\omega_{pe}t=3500$',linewidth=3)
    ax.plot(theta_plot,E_4300_f/E_4300_f[0],label=r'$\omega_{pe}t=4300$',linewidth=3)

    ax.legend(fontsize=20)
    ax.set_xlim(0,80)
    #plt.ylim(1e-4,100)
    ax.set_xlabel(r'$\theta ^\circ$',fontsize=26)
    ax.set_ylabel(r'$N(\theta) / N(0)$',fontsize=26)
    ax.tick_params(labelsize=20)
    plt.show()


if __name__ == '__main__':
    Ez_k_list, Ey_k_list,Ez_list,Ey_list,_ = load_phi()
    #k_main( Ez_k_list, Ey_k_list)
    theta_main(Ez_k_list, Ey_k_list,50)