import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

lz = 1.0
ly = 0.5

nz = 48
ny = 24

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

    
    return np.array(Ez_k_list), np.array(Ey_k_list), np.array(Ez_list), np.array(Ey_list), np.array(phi_list)

# def load_phi():
#     Ez_k_list=[]
#     Ey_k_list=[]
#     Ez_list = []
#     Ey_list = []
#     phi_list = []
#     for i in range(0,466):
#         fignum = str(i).zfill(4)
#         #filename = './Cori/mass25/rescheck/1/field/M25_E2_0_field_' + fignum + '.txt'
#         #filename = './massRatio/mass100/more_save/field/M100_E5_field_' + fignum + '.txt'
#         filename = './massRatio/mass100/E5_H2/field/M100_E5_field_' + fignum + '.txt'
#         phi = np.loadtxt(filename)
#         E_z, E_y = np.gradient(phi)
#         E_z = E_z/dz
#         E_y = E_y/dy
#         Ez_k = np.abs(np.fft.fftshift(np.fft.fftn(E_z)))
#         Ey_k = np.abs(np.fft.fftshift(np.fft.fftn(E_y)))
#         Ez_k_list.append(Ez_k)
#         Ey_k_list.append(Ey_k)
#         Ez_list.append(E_z)
#         Ey_list.append(E_y)
#         phi_list.append(phi)

    
#     return np.array(Ez_k_list), np.array(Ey_k_list), np.array(Ez_list), np.array(Ey_list), np.array(phi_list)


def k_omega_iaw():
    _,_,Ez0, Ey, phi = load_phi()

    Ez = np.transpose(Ez0[400:1200,:,0]) # 48 x 400
    Ezw = np.fft.rfft2(Ez) # 48 x 400
    Ezw2 = np.zeros_like(Ezw)
    for i in range(Ezw.shape[1]):
        Ezw2[:,i] = np.fft.fftshift(Ezw[:,i])
    absEzw = np.power(np.absolute(Ezw2),2)[:,:]  
    omegas = np.fft.rfftfreq(800,d=(400)/(800-1)) * 2.0 * np.pi
    ks = np.linspace(-24,23,48)/50

    k_list_25 = 2*np.pi*np.array([0, 0.3183, 1,      2,      4,     6,    8,     10,    12,     16,     18])
    gamma_list_25 =     np.array([0, 0.00143,0.00441,0.00833,0.0135,0.015,0.0141,0.0122,0.00973,0.00364,-0.0003])
    omega_list_25 =     np.array([0, 0.00835,0.0261, 0.051,  0.0946,0.128,0.153, 0.1718,0.187,  0.2122, 0.2237])
    f_25 = interp1d(k_list_25, gamma_list_25, kind='cubic')
    fw_25 = interp1d(k_list_25, omega_list_25, kind='cubic')
    k_sample_25 = np.arange(0.1,100,0.1)
    gamma_sample_25 = f_25(k_sample_25)
    omega_sample_25 = fw_25(k_sample_25)

    zz,yy = np.meshgrid(ks,omegas[:100],indexing='xy')
    zz = np.transpose(zz)
    yy = np.transpose(yy)
    absEzw = np.log(absEzw)
    absEzw[absEzw<-3] = -3
    #levels = [1e-9,1e-6,1e-3,1e-2,1e-1,1,3]

    fig     = plt.figure(figsize=(10.0,9.0))
    ax      = fig.add_axes([0.15, 0.15, 0.80, 0.76])
    ax.set_ylim(0,1.0)
    #plt.clim(vmin=-10,vmax=30)``
    levels = np.linspace(-3,3,21)
    pos = ax.contourf(zz,yy,absEzw[:,:100],levels=levels)
    ax.plot(-k_sample_25/2/np.pi/50, omega_sample_25,linewidth=5, color='red',linestyle='--',label=r'IAWs')
    ax.set_xlabel(r'$k\lambda_{De}/2\pi$',fontsize=36)
    ax.set_ylabel(r'$\omega \quad [\omega_{pe}]$',fontsize=36)
    ax.set_xlim(-20/50,20/50)
    ax.tick_params(labelsize = 32)
    l = ax.legend(fontsize=32)
    # for text in l.get_texts():
    #     text.set_color("white")
    plt.savefig('./Figures/figures_temp/iaw_ek_spectrum.jpg')
    plt.cla()

def k_omega_eaw():
    _,_,Ez0, Ey, phi = load_phi()

    k_list = 2*np.pi*np.arange(9,31,2)*0.1/50
    omega_list = np.array([0.0876485, 0.108983, 0.131577, 0.155846, 0.182402, 0.212296, 0.247897, 0.302664, 0.341438, 0.368753,0.385])
    fw = interp1d(k_list, omega_list, kind='cubic')
    k_sample = np.arange(9,27,0.1)*2*np.pi*0.1/50
    omega_sample = fw(k_sample)

    Ez = np.transpose(Ez0[3000:3600,:,0]) # 48 x 400
    Ezw = np.fft.rfft2(Ez) # 48 x 400
    Ezw2 = np.zeros_like(Ezw)
    for i in range(Ezw.shape[1]):
        Ezw2[:,i] = np.fft.fftshift(Ezw[:,i])
    absEzw = np.power(np.absolute(Ezw2),2)[:,:]  
    omegas = np.fft.rfftfreq(600,d=(300)/(600-1)) * 2.0 * np.pi
    ks = np.linspace(-24,23,48)/50
    #ks = np.linspace(-48,47,96)

    k_sample_25 = np.arange(0.1,24,0.1)/50

    zz,yy = np.meshgrid(ks,omegas[:100],indexing='xy')
    zz = np.transpose(zz)
    yy = np.transpose(yy)
    absEzw = np.log(absEzw)
    absEzw[absEzw<-3] = -3
    #levels = [1e-9,1e-6,1e-3,1e-2,1e-1,1,3]

    fig     = plt.figure(figsize=(10.0,9.0))
    ax      = fig.add_axes([0.15, 0.15, 0.80, 0.76])
    ax.set_ylim(0,1.0)
    #plt.clim(vmin=-10,vmax=30)
    levels = np.linspace(-3,3,21)
    pos = ax.contourf(zz,yy,absEzw[:,:100],levels=levels)
    ax.plot(-k_sample_25/2/np.pi, k_sample_25*1.2,linewidth=4, color='blue',linestyle='-.',label=r'$\omega/k = 1.2 v_{Te0}$')
    ax.plot(-k_sample/2/np.pi-0.01, omega_sample,linewidth=6, color='red',linestyle='--',label=r'EAWs')
    ax.set_xlabel(r'$k\lambda_{De}/2\pi$',fontsize=36)
    ax.set_ylabel(r'$\omega \quad [\omega_{pe}]$',fontsize=36)
    #ax.text(-2,0.7,r'$\frac{\omega}{k} = 1.2v_{Te}$',fontsize=36,color='white')
    #plt.show()
    ax.set_xlim(-0.4,0.4)
    ax.tick_params(labelsize = 32)
    l = ax.legend(fontsize=32)
    # for text in l.get_texts():
    #     text.set_color("white")
    #fig.colorbar(pos,ax=ax)
    #plt.show()
    plt.savefig('./Figures/figures_temp/eaw_ek_spectrum.jpg')
    #plt.cla()


if __name__ == '__main__':
    k_omega_eaw()
    #k_omega_iaw()