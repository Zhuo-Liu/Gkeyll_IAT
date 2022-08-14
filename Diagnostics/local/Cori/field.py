import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm

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
    plt.xlim(0,40)
    plt.show()
    plt.clf()

    return E_k

def load_phi():
    Ez_k_list=[]
    Ey_k_list=[]
    for i in range(100,400):
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
    
    return np.array(Ez_k_list), np.array(Ey_k_list)

def phi_plot():
    phi = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/field/M25_E2_3_field_0360.txt')
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    ax.contour3D(ZZ, YY, phi, 100)

    p = ax.plot_surface(ZZ, YY, phi, rstride=4, cstride=4, linewidth=0)

    # surface_plot with color grading and color bar
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    # cb = fig.colorbar(p, shrink=0.5)

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    cset = ax.contourf(ZZ, YY, phi, 30,  zdir='z', offset= -0.00015, cmap=matplotlib.cm.coolwarm)
    ax.set_zlim3d(-0.00015, 0.0003)
    plt.show()


def main():
    Ez_k_list, Ey_k_list = load_phi()
    #Ez_k_750 = Ez_k_list[]

if __name__ == '__main__':
    #main()
    phi_plot()