import skfuzzy as fuzz
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import os


# Semesta Pembicaraan
J_Kain = np.arange(13, 52, 1)
L_Pembuatan = np.arange(1, 92, 1)
M_otif = np.arange(10, 502, 1)
P_ewarnaan = np.arange(3, 22, 1)
H_Batik = np.arange(60, 1800, 1)

# Membership function


def memberFunction(kain, lama, motif, pewarna, hBatik):
    # Jenis Kain
    Kain_murah = fuzz.trimf(J_Kain, kain[0])
    Kain_sedang = fuzz.trimf(J_Kain, kain[1])
    Kain_mahal = fuzz.trimf(J_Kain, kain[2])
    # Lama Pembuatan
    Lama_Pembuatan_Cepat = fuzz.trimf(L_Pembuatan, lama[0])
    Lama_Pembuatan_Sedang = fuzz.trimf(L_Pembuatan, lama[1])
    Lama_Pembuatan_Lama = fuzz.trimf(L_Pembuatan, lama[2])
    # Jenis Kain
    Motif_Mudah = fuzz.trimf(M_otif, motif[0])
    Motif_Sedang = fuzz.trimf(M_otif, motif[1])
    Motif_Sulit = fuzz.trimf(M_otif, motif[2])
    # Pewarnaan
    Pewarnaan_Murah = fuzz.trimf(P_ewarnaan, pewarna[0])
    Pewarnaan_Sedang = fuzz.trimf(P_ewarnaan, pewarna[1])
    Pewarnaan_Mahal = fuzz.trimf(P_ewarnaan, pewarna[2])
    # Harga Jual
    Batik_Murah = fuzz.trimf(H_Batik, hBatik[0])
    Batik_Sedang = fuzz.trimf(H_Batik, hBatik[1])
    Batik_Mahal = fuzz.trimf(H_Batik, hBatik[2])

    return [Kain_murah, Kain_sedang, Kain_mahal], [Lama_Pembuatan_Cepat, Lama_Pembuatan_Sedang, Lama_Pembuatan_Lama], [Motif_Mudah, Motif_Sedang, Motif_Sulit], [Pewarnaan_Murah, Pewarnaan_Sedang, Pewarnaan_Mahal], [Batik_Murah, Batik_Sedang, Batik_Mahal]


def plotMember(kain, lama, motif, pewarnaan, hargaBatik):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(5, 7))
    ax0.plot(J_Kain, kain[0], 'b', linewidth=0.5, label="Murah")
    ax0.plot(J_Kain, kain[1], 'g', linewidth=0.5, label="Sedang")
    ax0.plot(J_Kain, kain[2], 'r', linewidth=0.5, label="Mahal")
    ax0.set_title("Harga Kain")
    ax0.legend()

    ax1.plot(L_Pembuatan, lama[0], 'b', linewidth=0.5, label="Cepat")
    ax1.plot(L_Pembuatan, lama[1], 'g', linewidth=0.5, label="Sedang")
    ax1.plot(L_Pembuatan, lama[2], 'r', linewidth=0.5, label="Lama")
    ax1.set_title("Lama Pembuatan")
    ax1.legend()

    ax2.plot(M_otif, motif[0], 'b', linewidth=0.5, label="Mudah")
    ax2.plot(M_otif, motif[1], 'g', linewidth=0.5, label="Sedang")
    ax2.plot(M_otif, motif[2], 'r', linewidth=0.5, label="Sulit")
    ax2.set_title("Motif")
    ax2.legend()
    plt.tight_layout()

    fig2, (ax3, ax4) = plt.subplots(nrows=2, figsize=(5, 7))
    ax3.plot(P_ewarnaan, pewarnaan[0], 'b', linewidth=0.5, label="Murah")
    ax3.plot(P_ewarnaan, pewarnaan[1], 'g', linewidth=0.5, label="Sedang")
    ax3.plot(P_ewarnaan, pewarnaan[2], 'r', linewidth=0.5, label="Mahal")
    ax3.set_title("Pewarnaan")
    ax3.legend()

    ax4.plot(H_Batik, hargaBatik[0], 'b', linewidth=0.5, label="Murah")
    ax4.plot(H_Batik, hargaBatik[1], 'g', linewidth=0.5, label="Sedang")
    ax4.plot(H_Batik, hargaBatik[2], 'r', linewidth=0.5, label="Mahal")
    ax4.set_title("Harga Batik")
    ax4.legend()
    plt.tight_layout()

    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    return fig, fig2


def fuzzifikasi(Kain, Lama, Motif, Pewarnaan, jK_input, lP_input, m_input, p_input):
    # Fungsi Keanggotaan Fuzzy
    dr_kain_murah = fuzz.interp_membership(J_Kain, Kain[0], jK_input)
    dr_kain_sedang = fuzz.interp_membership(J_Kain, Kain[1], jK_input)
    dr_kain_mahal = fuzz.interp_membership(J_Kain, Kain[2], jK_input)
    # print(f'Murah:{dr_kain_murah}',f'Sedang:{dr_kain_sedang}',f'Mahal:{dr_kain_mahal}')
    dr_LP_cepat = fuzz.interp_membership(L_Pembuatan, Lama[0], lP_input)
    dr_LP_sedang = fuzz.interp_membership(L_Pembuatan, Lama[1], lP_input)
    dr_LP_lama = fuzz.interp_membership(L_Pembuatan, Lama[2], lP_input)
    # print(f'Cepat:{dr_LP_cepat}',f'Sedang:{dr_LP_sedang}',f'Lama:{dr_LP_lama}')
    dr_M_mudah = fuzz.interp_membership(M_otif, Motif[0], m_input)
    dr_M_sedang = fuzz.interp_membership(M_otif, Motif[1], m_input)
    dr_M_sulit = fuzz.interp_membership(M_otif, Motif[2], m_input)
    # print(f'Mudah:{dr_M_mudah}',f'Sedang:{dr_M_sedang}',f'Sulit:{dr_M_sulit}')
    dr_P_murah = fuzz.interp_membership(P_ewarnaan, Pewarnaan[0], p_input)
    dr_P_sedang = fuzz.interp_membership(P_ewarnaan, Pewarnaan[1], p_input)
    dr_P_mahal = fuzz.interp_membership(P_ewarnaan, Pewarnaan[2], p_input)
    return dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal


def Rule(dataRule, dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal):
    # Convert DataFrame to NumPy array
    def jk_mapping_func(
        x): return dr_kain_murah if x == 'Murah' else dr_kain_sedang if x == 'Sedang' else dr_kain_mahal

    def lp_mapping_func(
        x): return dr_LP_cepat if x == 'Cepat' else dr_LP_sedang if x == 'Sedang' else dr_LP_lama

    def m_mapping_func(
        x): return dr_M_mudah if x == 'Mudah' else dr_M_sedang if x == 'Sedang' else dr_M_sulit
    def p_mapping_func(
        x): return dr_P_murah if x == 'Murah' else dr_P_sedang if x == 'Sedang' else dr_P_mahal

    # Apply mapping functions to the respective columns
    dataRule['Value_JK'] = dataRule['Harga Kain'].apply(jk_mapping_func)
    dataRule['Value_LP'] = dataRule['Lama Pembuatan'].apply(lp_mapping_func)
    dataRule['Value_M'] = dataRule['Motif'].apply(m_mapping_func)
    dataRule['Value_P'] = dataRule['Pewarnaan'].apply(p_mapping_func)


def Min(dataRule, harga):
    dataRule['MIN'] = np.minimum.reduce(
        [dataRule['Value_JK'], dataRule['Value_LP'], dataRule['Value_M'], dataRule['Value_P']])
    dataMin = []
    inMin = dataRule['MIN'].to_numpy()
    labelHarga = dataRule['Harga Batik'].to_numpy()

    for j in range(len(dataRule)):
        if labelHarga[j] == 'Murah':
            dataMin.append(np.fmin(inMin[j], harga[0]))
        elif labelHarga[j] == 'Sedang':
            dataMin.append(np.fmin(inMin[j], harga[1]))
        else:
            dataMin.append(np.fmin(inMin[j], harga[2]))

    return np.array(dataMin)


def Max(dataRule, harga):
    agregated = reduce(np.fmax, Min(dataRule, harga))
    return agregated


def valueMax(dataRule):
    temp_murah = 0
    temp_sedang = 0
    temp_mahal = 0
    for i in range(0, len(dataRule)):
        if (dataRule['Harga Batik'][i] == 'Murah'):
            if (dataRule['MIN'][i] > temp_murah):
                temp_murah = dataRule['MIN'][i]
        elif (dataRule['Harga Batik'][i] == 'Sedang'):
            if (dataRule['MIN'][i] > temp_sedang):
                temp_sedang = dataRule['MIN'][i]
        else:
            if (dataRule['MIN'][i] > temp_mahal):
                temp_mahal = dataRule['MIN'][i]
    Kom_Aturan = [temp_murah, temp_sedang, temp_mahal]
    return (Kom_Aturan)


def Defuzzifikasi(dataRule, harga):
    defuzzi = fuzz.defuzz(H_Batik, Max(dataRule, harga), 'centroid')
    return round(defuzzi)


def MAPE(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def EliminasiRule(dataRule, tempdatarule):
    unique_values = set(tempdatarule)
    new_list = list(unique_values)
    datarule_baru = pd.DataFrame()
    kolom_terpilih = ['Harga Kain', 'Lama Pembuatan',
                      'Motif', 'Pewarnaan', 'Harga Batik']
    for i in range(len(dataRule)):
        if (i in new_list):
            datarule_baru = pd.concat(
                [datarule_baru, dataRule.loc[[i], kolom_terpilih]], ignore_index=True)

    rule_terbaik = {
        'Harga Kain': datarule_baru['Harga Kain'],
        'Lama Pembuatan': datarule_baru['Lama Pembuatan'],
        'Motif': datarule_baru['Motif'],
        'Pewarnaan': datarule_baru['Pewarnaan'],
        'Harga Batik': datarule_baru['Harga Batik'],
    }

    df_rule_terbaik = pd.DataFrame(rule_terbaik)
    file_name = 'modelRule_best.xlsx'
    if os.path.exists(file_name):
        os.remove(file_name)
    df_rule_terbaik.to_excel(file_name, index=False)
