# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:34:36 2015

@author: JA Garzon
Edit 1: Sara Costa
Edit 2: Miguel Cruces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import time
import psutil
from scipy import stats

np.set_printoptions(formatter={'float': '{:.3f}'.format})
time_start = time.perf_counter()
# np.random.seed(11)

# =================================================================================================================== #
# ============================================== C O N S T A N T S ================================================== #
# =================================================================================================================== #

# Initial Configuration
final_prints = True
if_repr = True

# Physical Constants
c = 0.3  # [mm/ps]
sc = 1 / c  # [ps/mm] Slowness associated to the light celerity
mele = 0.511  # [MeV/c^2]
mmu = 105.6  # [MeV/c^2]
mpro = 938.3  # [MeV/c^2]

# Modifiable data
mass = mmu
kene = 1000  # MeV, Kinetic Energy
ene = mass * c + kene  # [MeV]
gamma = ene / (mass * c)
beta = np.sqrt(1 - 1 / (gamma * gamma))
betgam = beta * gamma
vini = beta * c  # [mm/ps] Initial Velocity
sini = 1 / vini  # [ps/mm]
pmom = betgam * mass  # [MeV/c^2]

ntrack = 3  # No. of tracks to generate
thmax = 10  # [deg] max theta
npar = 6  # No. of fit parameters
ndac = 3
mcut = 0.01  # Module of cut for iterations

# Initial values

sini = sc
tini = 1000
# ***
# xdini  = [mass/1000, ene/1000, beta, gamma];

# Detector Design
'''
- Rectangular detector with ncx * ncy rectangular electrodes
- It is assumed that the origin is in one edge of the detector

P L A N E S   D I S T R I B U T I O N

                                 [mm]   [mm]
T1 # -------------------------- # 1826      0  TOP

T2 # -------------------------- # 1304    522
T3 # -------------------------- #  924    902


T4 # -------------------------- #   87   1739  BOTTOM
                                     0         GROUND
'''
nplan = 4  # No. of planes
ncx = 12  # No. of cells in x direction
ncy = 10  # No. of cells in y direction
vz = np.array([1826, 1304, 924, 87])  # [mm] Planes position
vzi = vz[0] - vz  # [0, 522, 902, 1739] mm. Planes position relative to top plane
lenx = 1500  # [mm] Plane length in x direction
leny = 1200  # [mm] Plane length in y direction
lenz = vzi[-1] - vzi[0]  # [mm] Detector height (from top to bottom)
wcx = lenx / ncx  # [mm] Width of cell in x direction
wcy = leny / ncy  # [mm] Width of cell in y direction
wdt = 100  # [ps] Precision on time measurement

# Uncertainties
sigx = wcx / np.sqrt(12)  # [mm] Sigma X
sigy = wcy / np.sqrt(12)  # [mm] Sigma Y
sigt = 300  # [ps] Sigma T
wx = 1 / sigx ** 2
wy = 1 / sigy ** 2
wt = 1 / sigt ** 2
dt = 100  # Digitizer precision

# Measured data Vectors
vdx = np.zeros(nplan)
vdy = np.zeros(nplan)
vdt = np.zeros(nplan)

mtgen = np.zeros([ntrack, npar])  # Generated tracks matrix
mtrec = np.zeros([ntrack, npar])  # Reconstructed tracks matrix
vtrd = np.zeros(nplan * 3)  # Digitized tracks vector
mtrd = np.zeros([1, nplan * 3])  # Detector data Matrix
mErr = np.zeros([npar, npar])


# =================================================================================================================== #
# =================================== F U N C T I O N   D E F I N I T I O N S ======================================= #
# =================================================================================================================== #

# K   M A T R I C E S   A N D   V E C T O R   A


def set_K_tt(saeta, zi, mW):
    """
    Calculates the K k_mat Gain

    :param saeta: State vector
    :param zi: Height of the plane
    :param mW: Weights Matrix diagonal (WX, WY, WT)
    :return: K k_mat = mG.T * mW * mG.
    """
    mG = set_mG_tt(saeta, zi)  # mG: k_mat = partial m(s) / partial s
    mK = np.dot(mG.T, np.dot(mW, mG))
    return mK


def set_mG_tt(saeta, zi):
    """
    Jacobian k_mat

    :param saeta: State vector.
    :param zi: Height of the plane.
    :return: Jacobian Matrix
    """
    mG = np.zeros([ndac, npar])

    X0, XP, Y0, YP, T0, S0 = saeta
    ks = np.sqrt(1 + XP ** 2 + YP ** 2)
    ksi = 1 / ks

    mG[0, 0] = 1
    mG[0, 1] = zi
    mG[1, 2] = 1
    mG[1, 3] = zi
    mG[2, 1] = S0 * XP * zi * ksi
    mG[2, 3] = S0 * YP * zi * ksi
    mG[2, 4] = 1
    mG[2, 5] = ks * zi
    return mG


def v_g0_tt(vs, z):
    """
    Sets the g0 value

    :param vs: State vector (SAETA)
    :param z: Height of the current plane of the detector
    """
    vg0 = np.zeros(3)
    _, XP, _, YP, _, S0 = vs
    ks = np.sqrt(1 + XP ** 2 + YP ** 2)
    vg0[2] = - S0 * (XP ** 2 + YP ** 2) * z / ks
    return vg0


def set_vstat_tt(mG, mW, vdat, vg0):
    d_g0 = vdat - vg0
    va_out = np.dot(mG.T, np.dot(mW, d_g0))
    return va_out


# P L O T   F U N C T I O N S

def plot_saetas(vector, fig_id: int or str or None = None,
                plt_title=None, lbl: str = 'Vector', grids: bool = False,
                frmt_color: str = "green", frmt_marker: str = "--", prob_s=None):
    """
    Config Function for plot any SAETA with 6 parameters

    :param vector: The SAETA vector [X0, XP, Y0, YP, T0, S0]
    :param fig_id: Identification for the plot window
    :param plt_title:  Title for the plot
    :param lbl: Label for the SAETA
    :param grids: Set cell grids (higher CPU requirements, not recommendable)
    :param frmt_color: Format color for the SAETA representation
    :param frmt_marker: Format marker for the SAETA representation
    :param prob_s: value with alpha to fade SAETA.
    """
    # Plot configuration
    if fig_id is None:
        fig_id = 'State Vectors'
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    if plt_title is not None:
        ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, lenx])
    ax.set_ylim([0, leny])
    ax.set_zlim([vz[-1], vz[0]])

    # Unpack values
    x0, xp, y0, yp, t0, s0 = vector

    # Definition of variables
    z0 = vz[0]  # Detector Top Height
    z1 = vz[-1]  # Detector Bottom Height
    dz = z0 - z1  # Detector Height
    x1 = xp * dz
    y1 = yp * dz

    # Plot Vector
    x = np.array([x0, x0 + x1])
    y = np.array([y0, y0 + y1])
    z = np.array([z0, z1])
    if prob_s is not None:
        if 1 >= prob_s >= 0.9:
            frmt_color = "#FF0000"
        elif 0.9 > prob_s >= 0.6:
            frmt_color = "#FF5000"
        elif 0.6 > prob_s >= 0.3:
            frmt_color = "#FFA000"
        elif 0.3 > prob_s >= 0:
            frmt_color = "#FFF000"
        else:
            raise Exception(f"Ojo al dato: Prob = {prob_s}")
    ax.plot(x, y, z, linestyle=frmt_marker, color=frmt_color, label=lbl)
    ax.legend(loc='best')

    # Plot cell grid
    if grids:
        for zi in [-7000]:
            for yi in np.arange(-0.5, 10.5 + 1):
                for xi in np.arange(-0.5, 12.5 + 1):
                    plt.plot([-0.5, 12.5], [yi, yi], [zi, zi], 'k', alpha=0.1)
                    plt.plot([xi, xi], [-0.5, 10.5], [zi, zi], 'k', alpha=0.1)
    ax.legend(loc='best')
    # plt.show()


def plot_hit_ids(k_vec, fig_id: str = None, plt_title: str or None = None,
                 digi_trk: bool = True, cells: bool = True,
                 lbl: str = 'Digitized', frmt_color: str = "green", frmt_marker: str = ":"):
    """
    Config Function for plot any set of hits

    :param k_vec: Set of hits
    :param fig_id: Identification for the plot window
    :param plt_title: Title for the plot
    :param digi_trk: Set if reconstructed digitized track is shown
    :param cells: Set if hit cell squares are shown
    :param lbl: Label for the SAETA
    :param frmt_color: Format of color for the SAETA representation
    :param frmt_marker: Format of marker for the SAETA representation
    """
    # Set Plot - Initial Config
    if fig_id is None:
        fig_id = plt_title
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    if plt_title is not None:
        ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, lenx])
    ax.set_ylim([0, leny])
    ax.set_zlim([vz[-1], vz[0]])

    x = k_vec[np.arange(0, 12, 3)] * wcx
    y = k_vec[np.arange(1, 12, 3)] * wcy

    if cells:
        for ip in range(nplan):
            p = Rectangle(xy=(x[ip] - 0.5 * wcx, y[ip] - 0.5 * wcy),
                          width=wcx, height=wcy, alpha=0.5,
                          facecolor='#AF7AC5', edgecolor='#9B59B6', fill=True)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=vz[ip], zdir="z")

    if digi_trk:
        ax.plot(x, y, vz, linestyle=frmt_marker, color=frmt_color, label=lbl)
    ax.plot(x, y, vz, 'k.', alpha=0.9)

    ax.legend(loc='best')
    # plt.show()


def plot_detector(k_mat=None, fig_id=None, plt_title='Matrix Rays',
                  cells: bool = False, mtrack=None, mrec=None, prob_ary=None):
    """
    Config function for plot sets of hits and SAETAs

    :param k_mat: Matrix with all hits indices and times
    :param fig_id: Identification for the plot window
    :param plt_title: Title for the plot
    :param cells: Set if hit cell squares are shown
    :param mtrack: Array with all SAETAs generated
    :param mrec: Array with all SAETAs reconstructed
    :param prob_ary: Array with probabilities sorted by tracks order.
    """
    # Set Plot - Initial Config
    if fig_id is None:
        fig_id = plt_title
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, lenx])
    ax.set_ylim([0, leny])
    ax.set_zlim([vz[-1], vz[0]])

    # Plot Generated Tracks (SAETAs)
    if mtrack is not None:
        for trk in range(mtrack.shape[0]):
            plot_saetas(mtrack[trk], fig_id=fig_id,
                        lbl=f'Gene. {trk + 1}', frmt_color='#3498DB', frmt_marker='--')

    # Plot Digitized Tracks (Hits By Indices)
    if k_mat is not None:
        for trk in range(k_mat.shape[0]):
            plot_hit_ids(k_mat[trk], fig_id=fig_id,
                         lbl=f'Digi. {trk + 1}', frmt_color='#196F3D', frmt_marker=':', cells=cells)

    # Plot Reconstructed Tracks (SAETAs)
    if mrec is not None:
        for rec in range(mrec.shape[0]):
            if prob_ary is not None:
                plot_saetas(mrec[rec], fig_id=fig_id,
                            lbl=f'Reco. {rec + 1}', frmt_color='b', frmt_marker='-',
                            prob_s=prob_ary[rec])
            else:
                plot_saetas(mrec[rec], fig_id=fig_id,
                            lbl=f'Reco. {rec + 1}', frmt_color='b', frmt_marker='-')

    # plt.show()


# =================================================================================================================== #
# ====================================== T R A C K S   G E N E R A T I O N ========================================== #
# =================================================================================================================== #

ctmx = np.cos(np.deg2rad(thmax))
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

    # Director cosines
    cx = np.sin(tth) * np.cos(tph)
    cy = np.sin(tth) * np.sin(tph)
    cz = np.cos(tth)
    xp = cx / cz  # Projected slope on the X-Z plane
    yp = cy / cz  # Projected slope on the Y-Z plane

    # Coordenada por donde saldria la particula
    xzend = x0 + xp * lenz
    yzend = y0 + yp * lenz

    # Referimos la coordenada al centro del detector (xmid, ymid)
    xmid = xzend - (lenx / 2)
    ymid = yzend - (leny / 2)

    # Miramos si la particula ha entrado en el detector
    if (np.abs(xmid) < (lenx / 2)) and (np.abs(ymid) < (leny / 2)):
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
mtrd = np.delete(mtrd, 0, axis=0)

# =================================================================================================================== #
# ======================= T R A C K   A N A L Y S I S   A N D   R E C O N S T R U C T I O N ========================= #
# =================================================================================================================== #


for it in range(nt):
    vw = np.asarray([wx, wy, wt])
    mvw = np.zeros([3, 3])
    mvw[[0, 1, 2], [0, 1, 2]] = vw  # Fill diagonal with vw
    vs = [(lenx / 2), 0, (leny / 2), 0, 0, sc]
    vcut = 1
    cut = 0.1
    nit = 0
    while vcut > cut:
        mK = np.zeros([npar, npar])
        va = np.zeros(npar)
        so = np.zeros(nplan)

        for ip in range(nplan):
            zi = vzi[ip]
            ii = ip * 3
            dxi = mtrd[it, ii] * wcx
            dyi = mtrd[it, ii + 1] * wcy
            dti = mtrd[it, ii + 2]
            vdat = np.asarray([dxi, dyi, dti])

            mKi = set_K_tt(vs, zi, mvw)
            mG = set_mG_tt(vs, zi)
            vg0 = v_g0_tt(vs, zi)
            vai = set_vstat_tt(mG, mvw, vdat, vg0)

            mK = mK + mKi
            va = va + vai
            so += np.dot((vdat - vg0).T, np.dot(mvw, (vdat - vg0)))  # soi values

        mK = np.asmatrix(mK)
        mErr = mK.I

        va = np.asmatrix(va).T  # Vertical Measurement Vector
        vsol = np.dot(mErr, va)  # SEA equation

        sks = float(np.dot(vsol.T, np.dot(mK, vsol)))  # s'·K·s
        sa = float(np.dot(vsol.T, va))  # s'·a
        S = sks - 2 * sa + so  # S = s'·K·s - 2s'·a + So
        # print(f"S = sks - 2*sa + so = {sks:.3f} - 2*{sa:.3f} + {so:.3f} = {S:.3f}")

        DoF = nplan * ndac - npar  # Degrees of Freedom
        prob = stats.chi2.sf(S, DoF)
        vsol = np.asarray(vsol.T)[0]  # Make it a normal array again

        vdif = vs - vsol
        vdif = abs(vdif) / abs(vsol)  # (modulo de la diferencia)/(modulo del vector)
        vcut = max(vdif)
        vs = vsol
        nit += 1

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
# plt.savefig("Hist_dist.png", bbox_inches='tight')

plt.figure(2)
n2, bins2, patches2 = plt.hist(distanciax, bins=20, alpha=1, linewidth=1)
plt.title('Distancia entre puntos incidencia en X y reconstruidos en X')
plt.grid(True)
# plt.savefig("Hist_distX.png", bbox_inches='tight')

plt.figure(3)
n3, bins3, patches3 = plt.hist(distanciay, bins=20, alpha=1, linewidth=1)
plt.title('Distancia entre puntos incidencia en Y y reconstruidos en Y')
plt.grid(True)
# plt.savefig("Hist_distY.png", bbox_inches='tight')

# Scatter plot
plt.figure(4)
plt.scatter(distanciax, distanciay)  # , marker='o')  # , s=1)
plt.title('Scatter plot distX vs distY')
plt.grid(True)
# plt.savefig("Scatterplot_XY.png", bbox_inches='tight')

plt.figure(5)
plt.scatter(distanciax, distanciaxp)  # , marker='.')  # , s=1)
plt.title('Scatter plot distX vs distX´ ')
plt.grid(True)
# plt.savefig("Scatterplot_XXP.png", bbox_inches='tight')

plt.figure(6)
plt.scatter(distanciay, distanciayp)  # , marker='X')  # , s=1)
plt.title('Scatter_plot distY vs distY´ ')
plt.grid(True)
# plt.savefig("Scatterplot_YYP.png", bbox_inches='tight')


if if_repr:
    # prob_tt = mtrec[:, -1]
    # prob_kf = m_stat[:, -1]
    # k_mat_gene = mdat
    plot_detector(fig_id=f"Genedigitana {ntrack}", plt_title=f"Tim Track", cells=True,
                  k_mat=mtrd, mtrack=mtgen, mrec=mtrec)  # , prob_ary=prob_tt)
    plt.show()

if final_prints:
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

    print('Distance between GENERATED and RECONSTRUCTED tracks')
    # Mean
    s = 0
    for i in range(len(n)):
        s += n[i] * ((bins[i] + bins[i + 1]) / 2)
    mean = s / np.sum(n)

    # Standard Deviation
    t = 0
    for i in range(len(n)):
        t += n[i] * (bins[i] - mean) ** 2
    std = np.sqrt(t / np.sum(n))

    print(f'+-----------------------+')
    print(f'| Mean: {mean:.3f} mm      |')
    print(f'| Std. dev.: {std:.3f} mm |')
    print(f'+-----------------------+')

    print('\nError Matrix:')
    for row in mRed:
        for col in row:
            print(f"{col:8.3f}", end=" ")
        print("")

    time_elapsed = (time.perf_counter() - time_start)
    print('\n+----- Machine Stats -----+')
    print(f'| Computing time: {time_elapsed:.3f} s |')
    print(f'| CPU usage {psutil.cpu_percent():.1f} %\t\t  |')
    print('+-------------------------+')
