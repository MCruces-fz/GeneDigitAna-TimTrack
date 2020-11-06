# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:34:36 2015

@author: jag
Edit 1: Sara Costa
Edit 2: Miguel Cruces
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
from scipy import stats

np.set_printoptions(formatter={'float': '{:.3f}'.format})
time_start = time.perf_counter()
np.random.seed(11)


# =================================================================================================================== #
# =================================== F U N C T I O N   D E F I N I T I O N S ======================================= #
# =================================================================================================================== #

# K   M A T R I C E S   A N D   V E C T O R   A

def mKpads4(z, wx, wy, wt):
    """
    K matrix for a pad plane with 4 parameters: X0, XP, Y0, YP

    :param z:
    :param wx:
    :param wy:
    :param wt:
    :return:
    """
    mK = np.zeros([npar, npar])
    mK[0, 0] = wx
    mK[0, 1] = wx * z
    mK[1, 1] = wx * z * z
    mK[1, 0] = mK[0, 1]
    mK[2, 2] = wy
    mK[2, 3] = wy * z
    mK[3, 3] = wy * z * z
    mK[3, 2] = mK[2, 3]
    return mK



def v_g0_pads(vs, z):
    vg0 = np.zeros(3)
    xp = vs[1]
    yp = vs[3]
    s0 = vs[5]
    ks2 = 1 + xp * xp + yp * yp
    vg0[2] = -((xp * xp + yp * yp) * s0 * z) / np.sqrt(ks2)

    return vg0


def m_K_a_pads(vs, z, vw, vdat):
    """
    Function that returns: K matrix and vector a for a pad plane.
    6 parameters version: X0, XP, Y0, YP, T0, S0

    :param vs:
    :param z:
    :param vw:
    :param vdat:
    :return:
    """
    mk = np.zeros([npar, npar])
    vx = np.zeros(npar)
    xp = vs[1]
    yp = vs[3]
    # t0   = vs[4]
    s0 = vs[5]
    ks2 = 1 + xp * xp + yp * yp  # slope factor
    wx = vw[0]
    wy = vw[1]
    wt = vw[2]
    dx = vdat[0]
    dy = vdat[1]
    dt = vdat[2]

    vx[0] = wx * dx
    vx[1] = z * (wx * dx + wt * xp * s0 * (dt * (1 / np.sqrt(ks2)) + z * (1 / ks2) * (xp ** 2 + yp ** 2) * s0))
    vx[2] = wy * dy
    vx[3] = z * (wy * dy + wt * yp * s0 * (dt * (1 / np.sqrt(ks2)) + z * (1 / ks2) * (xp ** 2 + yp ** 2) * s0))
    vx[4] = wt * (dt + z * (1 / np.sqrt(ks2)) * s0 * (xp ** 2 + yp ** 2))
    vx[5] = z * wt * ks * (dt + z * (1 / np.sqrt(ks2)) * (xp ** 2 + yp ** 2) * s0)

    mk[0, 0] = wx
    mk[0, 1] = z * wx
    mk[1, 1] = z ** 2 * (wx + wt * (1 / ks2) * xp * xp * s0 * s0)
    mk[1, 3] = z ** 2 * wt * (1 / ks2) * xp * s0
    mk[1, 4] = z * wt * (1 / np.sqrt(ks2)) * xp * s0
    mk[1, 5] = z ** 2 * wt * xp * s0
    mk[2, 2] = wy
    mk[2, 3] = z * wy
    mk[3, 3] = z ** 2 * (wy + wt * (1 / ks2) * yp * yp * s0 * s0)
    mk[3, 4] = z * wt * (1 / ks2) * yp * s0
    mk[3, 5] = z ** 2 * wt * yp * s0
    mk[4, 4] = wt
    mk[4, 5] = z * wt * np.sqrt(ks2)
    mk[5, 5] = z ** 2 * wt * ks2

    # Por ser simetrica, mK=mK' (traspuesta)
    mk = mk + mk.T - np.diag(mk.diagonal())

    return mk, vx


# =================================================================================================================== #
# ============================================== C O N S T A N T S ================================================== #
# =================================================================================================================== #

# Physical Constants
c = 0.3  # [mm/ps]
sc = 1 / c  # lentitud asociada a la velocidad de la luz
mele = 0.511
mmu = 105.6
mpro = 938.3

# Modifiable data
masa = mmu
kene = 1000  # MeV, energia cinetica
ene = masa + kene
gamma = ene / masa
beta = np.sqrt(1 - 1 / (gamma * gamma))
betgam = beta * gamma
vini = beta * c  # velocidad inicial
sini = 1 / vini
pmom = betgam * masa

ntrack = 10  # num. de trazas a generar
thmax = 10  # max theta en grados
npar = 6  # numero de parametros a ajustar
mcut = 0.01  # modulo de corte para la iteracion

# Initial values

sini = sc
tini = 1000
# ***
# xdini  = [masa/1000, ene/1000, beta, gamma];


# Detector design

# Detector rectangular con ncx*ncy electrodos rectangulares
# Asumimos el origen de coordenadas en una esquina del detector

nplan = 4  # num. de planos
ncx = 12  # num. celdas en x
ncy = 10  # num. celdas en y
vzi = [0, 600, 900, 1800]  # posicion de los planos
lenx = 1500  # longitud en x
leny = 1200  # longitud en y
wcx = lenx / ncx  # anchura de la celda en x
wcy = leny / ncy  # anchura de la celda en y
wdt = 100

# Uncertainty
sigx = (1 / np.sqrt(12)) * wcx
sigy = (1 / np.sqrt(12)) * wcy
sigt = 300  # [ps]
wx = 1 / sigx ** 2
wy = 1 / sigy ** 2
wt = 1 / sigt ** 2
dt = 100  # precision del digitalizador

# Measured data Vectors
vdx = np.zeros(nplan)
vdy = np.zeros(nplan)
vdt = np.zeros(nplan)

mtgen = np.zeros([ntrack, npar])  # matriz de trazas generadas
mtrec = np.zeros([ntrack, npar])  # matriz de trazas reconstruidas
vtrd = np.zeros(nplan * 3)  # vector de trazas digitalizacion
mtrd = np.zeros([1, nplan * 3])  # matriz de datos del detector
mErr = np.zeros([npar, npar])

# =================================================================================================================== #
# ====================================== T R A C K S   G E N E R A T I O N ========================================== #
# =================================================================================================================== #

ctmx = np.cos(np.deg2rad(thmax))  # coseno de theta_max
lenz = vzi[nplan - 1] - vzi[0]
it = 0

for i in range(ntrack):
    # Distribucion uniforme en cos(theta) y phi
    rcth = 1 - np.random.random() * (1 - ctmx)
    tth = np.arccos(rcth)  # theta
    tph = np.random.random() * 2 * np.pi  # phi

    x0 = np.random.random() * lenx
    y0 = np.random.random() * leny
    t0 = tini
    s0 = sini

    cx = np.sin(tth) * np.cos(tph)  # cosenos directores
    cy = np.sin(tth) * np.sin(tph)
    cz = np.cos(tth)
    xp = cx / cz  # pendiente proyectada en el plano X-Z
    yp = cy / cz  # pendiente proyectada en el plano Y-Z
    '''
    #Parche para simular una traza concreta
    x0 = 1000
    xp = 0.1
    y0 = 600
    yp = -0.1
    '''

    # Coordenada por donde saldria la particula
    xzend = x0 + xp * lenz
    yzend = y0 + yp * lenz

    # Referimos la coordenada al centro del detector (xmid, ymid)
    xmid = xzend - (lenx / 2)
    ymid = yzend - (leny / 2)

    # Miramos si la particula ha entrado en el detector
    if ((np.abs(xmid) < (lenx / 2)) and (np.abs(ymid) < (leny / 2))):
        mtgen[it, :] = [x0, xp, y0, yp, t0, s0]
        it = it + 1
    else:
        continue
nt = it
# Borro las lineas de ceros (en las que la particula no entro en el detector)
mtgen = mtgen[~(mtgen == 0).all(1)]

# vstrk = [x0, xp, y0, yp, tini, sini]

# =================================================================================================================== #
# =========================================== D I G I T I Z A T I O N =============================================== #
# =================================================================================================================== #


nx = 0
for it in range(nt):
    x0 = mtgen[it, 0]
    xp = mtgen[it, 1]
    y0 = mtgen[it, 2]
    yp = mtgen[it, 3]
    # dz = np.cos(th)

    it = 0
    for ip in range(nplan):
        zi = vzi[ip]
        xi = x0 + xp * zi
        yi = y0 + yp * zi
        ks = np.sqrt(1 + xp * xp + yp * yp)
        ti = tini + ks * sc * zi
        # Indices de posicion de las celdas impactadas
        kx = np.int((xi + (wcx / 2)) / wcx)
        ky = np.int((yi + (wcy / 2)) / wcy)
        kt = np.int((ti + (dt / 2)) / dt) * dt
        xic = kx * wcx + (wcx / 2)
        yic = ky * wcy + (wcy / 2)
        vxyt = np.asarray([kx, ky, kt])
        vtrd[it:it + 3] = vxyt[0:3]
        it = it + 3
    mtrd = np.vstack((mtrd, vtrd))
    nx = nx + 1
mtrd = np.delete(mtrd, (0), axis=0)

# =================================================================================================================== #
# ======================= T R A C K   A N A L Y S I S   A N D   R E C O N S T R U C T I O N ========================= #
# =================================================================================================================== #

vw = np.asarray([wx, wy, wt])
mvw = np.zeros([3, 3])
np.fill_diagonal(mvw, vw)
vsim = np.asarray([x0, xp, y0, yp, tini, sini])
vsini = [(lenx / 2), 0, (leny / 2), 0, 0, sc]

vcut = 1
cut = 0.1
nit = 0
vs = vsini

for it in range(nt):
    while (vcut > cut):  # Iteracion
        mK = np.zeros([npar, npar])
        va = np.zeros(npar)
        so = np.zeros(nplan)

        for ip in range(nplan):
            zi = vzi[ip]
            ii = ip * 3
            dxi = mtrd[it, ii] * wcx - (wcx / 2)  # data
            dyi = mtrd[it, ii + 1] * wcy - (wcy / 2)
            dti = mtrd[it, ii + 2]
            vdx[ip] = dxi
            vdy[ip] = dyi
            vdt[ip] = dti
            vdat = np.asarray([dxi, dyi, dti])

            mKi, vai = m_K_a_pads(vs, zi, vw, vdat)
            mK = mK + mKi
            va = va + vai
            vg0 = v_g0_pads(vs, vzi[ip])
            so[ip] = np.dot((vdat - vg0).T, np.dot(mvw, (vdat - vg0)))

        mK = np.asmatrix(mK)
        mErr = mK.I

        vsol = np.dot(mErr, va)  # SEA equation
        vsol = np.array(vsol)[0]

        vdif = vs - vsol
        vdif = abs(vdif) / abs(vsol)  # (modulo de la diferencia)/(modulo del vector)
        vcut = max(vdif)
        vs = vsol
        nit = nit + 1

    sk = np.dot(np.dot(vsol.T, mK), vsol)
    sa = np.dot(vsol.T, va)
    so = np.sum(so)
    S = sk - 2 * sa + so
    prob = stats.chi2.sf(S, 6)

    mtrec[it, :] = vsol
mtrec = mtrec[~(mtrec == 0).all(1)]

# Calculo distancias entre puntos de incidencia y reconstruidos

distanciax = np.zeros([nt, 1])
distanciay = np.zeros([nt, 1])
distanciaxp = np.zeros([nt, 1])
distanciayp = np.zeros([nt, 1])
distancia = np.zeros([nt, 1])
for i in range(nt):
    for j in range(6):
        distanciax = abs(mtrec[:, 0] - mtgen[:, 0])
        distanciay = abs(mtrec[:, 2] - mtgen[:, 2])
        distanciaxp = abs(mtrec[:, 1] - mtgen[:, 1])
        distanciayp = abs(mtrec[:, 3] - mtgen[:, 3])
        distancia = np.sqrt(distanciax ** 2 + distanciay ** 2)

# Hago un histograma con esas distancias
plt.figure(1)
n, bins, patches = plt.hist(distancia, bins=20, alpha=1, linewidth=1)
plt.title('Distancia entre puntos incidencia y reconstruidos')
plt.grid(True)
# plt.show()
# plt.savefig("Hist_dist.png", bbox_inches='tight')

plt.figure(2)
n2, bins2, patches2 = plt.hist(distanciax, bins=20, alpha=1, linewidth=1)
plt.title('Distancia entre puntos incidencia en X y reconstruidos en X')
plt.grid(True)
# plt.show()
# plt.savefig("Hist_distX.png", bbox_inches='tight')

plt.figure(3)
n3, bins3, patches3 = plt.hist(distanciay, bins=20, alpha=1, linewidth=1)
plt.title('Distancia entre puntos incidencia en Y y reconstruidos en Y')
plt.grid(True)
# plt.show()
# plt.savefig("Hist_distY.png", bbox_inches='tight')

# Calculo la media
s = 0
for i in range(len(n)):
    s += n[i] * ((bins[i] + bins[i + 1]) / 2)
mean = s / np.sum(n)
print('La media es', mean)

# Calculo la desviacion estandard
t = 0
for i in range(len(n)):
    t += n[i] * (bins[i] - mean) ** 2
std = np.sqrt(t / np.sum(n))
print('La desviacion tipica es', std)

# Scatter plot
plt.figure(4)
plt.scatter(distanciax, distanciay, s=1)
plt.title('Scatter plot distX vs distY')
plt.grid(True)
# plt.show()
# plt.savefig("Scatterplot_XY.png", bbox_inches='tight')

plt.figure(5)
plt.scatter(distanciax, distanciaxp, s=1)
plt.title('Scatter plot distX vs distX´ ')
plt.grid(True)
# plt.show()
# plt.savefig("Scatterplot_XXP.png", bbox_inches='tight')

plt.figure(6)
plt.scatter(distanciay, distanciayp, s=1)
plt.title('Scatter_plot distY vs distY´ ')
plt.grid(True)
# plt.show()
# plt.savefig("Scatterplot_YYP.png", bbox_inches='tight')

# Matriz de error reducida

sigp1 = np.sqrt(mErr[0, 0])
sigp2 = np.sqrt(mErr[1, 1])
sigp3 = np.sqrt(mErr[2, 2])
sigp4 = np.sqrt(mErr[3, 3])
sigp5 = np.sqrt(mErr[4, 4])
sigp6 = np.sqrt(mErr[5, 5])

cor12 = mErr[0, 1] / (sigp1 * sigp2)
cor13 = mErr[0, 2] / (sigp1 * sigp3)
cor14 = mErr[0, 3] / (sigp1 * sigp4)
cor15 = mErr[0, 4] / (sigp1 * sigp5)
cor16 = mErr[0, 5] / (sigp1 * sigp6)
cor23 = mErr[1, 2] / (sigp2 * sigp3)
cor24 = mErr[1, 3] / (sigp2 * sigp4)
cor25 = mErr[1, 4] / (sigp2 * sigp5)
cor26 = mErr[1, 5] / (sigp2 * sigp6)
cor34 = mErr[2, 3] / (sigp3 * sigp4)
cor35 = mErr[2, 4] / (sigp3 * sigp5)
cor36 = mErr[2, 5] / (sigp3 * sigp6)
cor45 = mErr[3, 4] / (sigp4 * sigp5)
cor46 = mErr[3, 5] / (sigp4 * sigp6)
cor56 = mErr[4, 5] / (sigp5 * sigp6)

mRed = np.array([[sigp1, cor12, cor13, cor14, cor15, cor16],
                 [0, sigp2, cor23, cor24, cor25, cor26],
                 [0, 0, sigp3, cor34, cor35, cor36],
                 [0, 0, 0, sigp4, cor45, cor46],
                 [0, 0, 0, 0, sigp5, cor56],
                 [0, 0, 0, 0, 0, sigp6]])

for row in mRed:
    for col in row:
        print("{:8.3f}".format(col), end=" ")
    print("")

time_elapsed = (time.perf_counter() - time_start)
print('Computing time:', time_elapsed)

print('CPU usage', psutil.cpu_percent())
