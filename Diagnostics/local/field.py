import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm
from zmq import EVENT_HANDSHAKE_FAILED_AUTH

# x = np.arange(0,2,0.01)
# phi1 = np.cos(2*np.pi*x)
# phi2 = np.cos(2*2*np.pi*x)
# phi = phi1+phi2
# Ex = np.gradient(phi)/0.01/(2*np.pi)
# Ex2 = Ex*Ex

# Exk = np.fft.fftshift(np.fft.fft(Ex))
# Ex_freq = np.fft.fftshift(np.fft.fftfreq(x.size,0.01))

# plt.grid(linestyle=":")
# plt.plot(Ex_freq, np.abs(Exk))
# plt.xlim(-25,25)
# plt.show()

# plt.plot(x,phi,label='phi')
# # plt.plot(x,Ex,label='E')
# # plt.plot(x,Ex2)
# plt.legend()
# plt.show()

lz = 1.0
ly = 0.5

nz = 96
ny = 48

dz = lz/nz
dy = ly/ny

z_plot = np.linspace(0,lz,nz)
y_plot = np.linspace(0,ly,ny)
ZZ, YY = np.meshgrid(z_plot, y_plot, indexing= 'xy')

kz_plot   = 2.0*np.pi*np.linspace(-int(nz/2), int(nz/2-1), nz)/lz
ky_plot  = 2.0*np.pi*np.linspace(-int(ny/2), int(ny/2-1), ny)/ly
K_z, K_y = np.meshgrid(kz_plot, ky_plot, indexing = 'xy')
K_z = np.transpose(K_z)
K_y = np.transpose(K_y)

phi = np.loadtxt('/Users/liuzhuo/Desktop/Gkeyll_IAT/Diagnostics/local/Cori/mass25/rescheck/4/field/M25_E2_3_field_0160.txt')

E_z, E_y = np.gradient(phi)
E_z = E_z/dz
E_y = E_y/dy

Ez_k = np.abs(np.fft.fftshift(np.fft.fftn(E_z)))
Ey_k = np.abs(np.fft.fftshift(np.fft.fftn(E_y)))

def plot_phi():
    plt.contourf(ZZ, YY, np.transpose(phi))
    plt.xlabel(r'$z$',fontsize=30)
    plt.ylabel(r'$y$',fontsize=30)
    #plt.tick_params(labelsize = 26)
    plt.colorbar()
    plt.show()

def plot_E():
    fig, axs = plt.subplots(1,2,figsize=(12, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace =.1)
    axs = axs.ravel()

    norm = colors.LogNorm(0.1,10)
    sm= plt.cm.ScalarMappable(cmap='jet',norm=norm)
    #sm.set_array([])

    pos0 = axs[0].contourf(K_z/2/np.pi, K_y/2/np.pi, Ez_k, 10,camp=sm)
    axs[0].set_xlabel(r'$k_z/2\pi$', fontsize=24)
    axs[0].set_ylabel(r'$k_y/2\pi$', fontsize=24, labelpad=-1)
    axs[0].set_xlim(0,16)
    axs[0].set_ylim(0,12)
    axs[0].set_title(r'$E_{xk}$', fontsize=20)
    axs[0].tick_params(labelsize = 20)
    cbar = fig.colorbar(sm, ax=axs[0])
    cbar.ax.tick_params(labelsize=20)

    pos1 = axs[1].contourf(K_z/2/np.pi, K_y/2/np.pi, Ey_k)
    axs[1].set_xlabel(r'$k_z/2\pi$', fontsize=24)
    axs[1].set_ylabel(r'$k_y/2\pi$', fontsize=24, labelpad=-1)
    axs[1].set_xlim(0,16)
    axs[1].set_ylim(0,12)
    axs[1].set_title(r'$E_{yk}$', fontsize=20)
    axs[1].tick_params(labelsize = 20)
    cbar = fig.colorbar(pos1, ax=axs[1])
    cbar.ax.tick_params(labelsize=20)   


    fig.tight_layout()
    plt.show()

    # plt.xlim(-16,16)
    # plt.ylim(-16,16)
    # #plt.tick_params(labelsize = 26)
    # plt.colorbar()
    # plt.show()

def theta_dependence(N):
    E_theta = np.zeros(N)
    for z in range(nz):
        for y in range(ny):
            ez = E_z[z,y]
            ey = E_y[z,y]
            theta = np.abs(np.arctan(ey/ez))
            num  = int(theta/(np.pi/2/N))
            E_theta[num] = E_theta[num] + ez**2 + ey**2

    plt.plot(np.arange(N)*90/N,E_theta)
    plt.show()

    return E_theta

def k_dependence(N):
    E_k = np.zeros(48*N)
    for i in range(nz):
        for j in range(ny):
            k_square = (i-int(nz/2))**2 + (j-int(ny/2))**2
            k_star = np.sqrt(k_square)
            if k_star < 48:
                decimal = k_star - int(k_star)
                app_inx = int(decimal * N)
                Ek_square = Ez_k[i,j]**2 + Ey_k[i,j]**2
                E_k[N*int(k_star)+app_inx] += Ek_square
    
    k_list = np.arange(0.6,48,0.1)*np.pi*2
    N_k = 1/(k_list*k_list*k_list) * np.log(1/k_list/0.02) 

    plt.plot(np.arange(48*N)/N,E_k/450)
    plt.plot(k_list,N_k)
    plt.xlim(0,12)
    plt.show()
    plt.clf()

    return E_k

# def k_theta_dependence():
#     E_k_theta = np.zeros((48,24))
#     for i in range(nz):
#         for j in range(ny):
#             kz = i-int(nz/2)
#             ky = j-int(ny/2)
#             k_square = kz**2 + ky**2
#             k_star = int(np.sqrt(k_square))
#             if k_star < 48:
#                 Ek_square = Ez_k[kz,ky]**2 + Ey_k[kz,ky]**2

#                 if kz ==0:
#                     theta = np.pi/2
#                 else:
#                     theta = np.abs(np.arctan(ky/kz))

#                 theta_num = int(theta/(np.pi/2/24))
#                 if theta_num == 24:
#                     theta_num=23
#                 E_k_theta[k_star, theta_num] += Ek_square 
    
#     E_k = np.sum(E_k_theta,axis=1)
#     E_theta = np.sum(E_k_theta, axis=0)

#     return E_k_theta, E_k, E_theta

def test():
    Ex = np.zeros((100,100))
    Ey = np.zeros((100,100))
    E = np.zeros((100,100))
    for x in np.arange(0,1,0.01):
        for y in np.arange(0,1,0.01):             
            Ex[int(x*100),int(y*100)] = np.cos(2*np.pi*(10*x+y))
            Ey[int(x*100),int(y*100)] = np.cos(2*np.pi*(10*x+y))

            #E = np.cos(2*np.pi/np.sqrt(2)*np.arange(0,10*np.sqrt(2),0.01))

    Ekx = np.fft.fftshift(np.fft.fftn(Ex))
    Eky = np.fft.fftshift(np.fft.fftn(Ey))
    #Ek = np.fft.fftshift(np.fft.fftn(E))

    Ex_freq = np.fft.fftshift(np.fft.fftfreq(Ex.shape[0],0.01))
    Ey_freq = np.fft.fftshift(np.fft.fftfreq(Ex.shape[1],0.01))

    KKX, KKY = np.meshgrid(Ex_freq, Ey_freq, indexing = 'xy')

    plt.contourf(KKX,KKY,np.transpose(np.abs(Eky)))
    plt.xlim(-16,16)
    plt.ylim(-16,16)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    #plot_E()

    #theta_dependence(15)

    E_k = k_dependence(2)

    # plt.plot(np.arange(48*100)/100,E_k)
    # plt.xlim(0,10)
    # plt.show()

    # plt.plot(np.arange(48),E_k)
    # #plt.xlim(0,20)
    # plt.show()

    #theta_dependence2()