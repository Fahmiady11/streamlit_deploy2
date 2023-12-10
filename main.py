import os
import ast
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import time
import component.fuzzyMamdani.fuzzyMamdani as fuzzyMamdani
import component.algoritmaGenetika.algoritmaGenetika as algoritmaGenetika
# data
data = pd.read_excel('assets/dataBatik.xlsx', sheet_name='dataset')
dataTrain = pd.read_excel('assets/dataBatik.xlsx', sheet_name='train')
dataTest = pd.read_excel('assets/dataBatik.xlsx', sheet_name='test')
dataTrain50 = pd.read_excel('assets/dataBatik.xlsx', sheet_name='train50')
dataTest50 = pd.read_excel('assets/dataBatik.xlsx', sheet_name='test50')

dataRule = pd.read_excel('assets/rule_data.xlsx')

st.markdown(
    """
    <style>
       
        table th {
            background-color: #f0f0f0;  
            color: #000000;
            border: 1px solid black;
            

        }

        table td {
            background-color: #f0f0f0;  
            color: #000000;
            border: 1px solid black;
        }

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Batik CV NARAYA", ["Home", "Dataset", 'Rule Aturan', 'Fuzzy Mamdani', 'Fuzzy Mamdani & GA'],
                           icons=['house', 'database', 'database', 'journal-code', 'journal-code'], menu_icon="collection", default_index=0)


def mainFuzzy(status):
    if (status == "train"):
        jk_Input = dataTrain['Harga Kain']
        lp_Input = dataTrain['Lama Pembuatan']
        m_Input = dataTrain['Motif']
        p_Input = dataTrain['Pewarnaan']
        aktual = dataTrain['Harga Aktual'].to_numpy()
        dataRule = pd.read_excel('assets/rule_data.xlsx')
    elif (status == "test"):
        jk_Input = dataTest['Harga Kain']
        lp_Input = dataTest['Lama Pembuatan']
        m_Input = dataTest['Motif']
        p_Input = dataTest['Pewarnaan']
        aktual = dataTest['Harga Aktual'].to_numpy()
        dataRule = pd.read_excel('modelRule_best.xlsx')
    elif (status == "train50"):
        jk_Input = dataTrain50['Harga Kain']
        lp_Input = dataTrain50['Lama Pembuatan']
        m_Input = dataTrain50['Motif']
        p_Input = dataTrain50['Pewarnaan']
        aktual = dataTrain50['Harga Aktual'].to_numpy()
        dataRule = pd.read_excel('assets/rule_data.xlsx')
    else:
        jk_Input = dataTest50['Harga Kain']
        lp_Input = dataTest50['Lama Pembuatan']
        m_Input = dataTest50['Motif']
        p_Input = dataTest50['Pewarnaan']
        aktual = dataTest50['Harga Aktual'].to_numpy()
        dataRule = pd.read_excel('modelRule_best.xlsx')

    # InputMemberFunction
    Kain_murah = [13, 13, 26]
    Kain_sedang = [24, 31.5, 39]
    Kain_mahal = [37, 52, 52]
    #
    Lama_Pembuatan_Cepat = [1, 1, 31]
    Lama_Pembuatan_Sedang = [29, 45.5, 62]
    Lama_Pembuatan_Lama = [60, 92, 92]
    #
    Motif_Mudah = [10, 10, 165]
    Motif_Sedang = [160, 247.5, 335]
    Motif_Sulit = [330, 502, 502]
    #
    Pewarnaan_Murah = [3, 3, 9]
    Pewarnaan_Sedang = [8, 13, 18]
    Pewarnaan_Mahal = [16, 22, 22]
    #
    Batik_Murah = [60, 60, 150]
    Batik_Sedang = [100, 350, 600]
    Batik_Mahal = [500, 1800, 1800]

    fuzzifikasi = []
    dataMax = []
    dataPredAkt = []
    prediksi = []
    tempdatarule = []
    kain, lama, motif, pewarnaan, hargaBatik = fuzzyMamdani.memberFunction([Kain_murah, Kain_sedang, Kain_mahal], [Lama_Pembuatan_Cepat, Lama_Pembuatan_Sedang, Lama_Pembuatan_Lama], [
                                                                           Motif_Mudah, Motif_Sedang, Motif_Sulit], [Pewarnaan_Murah, Pewarnaan_Sedang, Pewarnaan_Mahal], [Batik_Murah, Batik_Sedang, Batik_Mahal])
    fig, fig2 = fuzzyMamdani.plotMember(
        kain, lama, motif, pewarnaan, hargaBatik)
    for j in range(len(jk_Input)):
        dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal = fuzzyMamdani.fuzzifikasi(
            kain, lama, motif, pewarnaan, jk_Input[j], lp_Input[j], m_Input[j], p_Input[j])
        fuzzifikasi.append([dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang,
                           dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal])
        fuzzyMamdani.Rule(dataRule, dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang,
                          dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal)
        try:
            hasilDefuzz = fuzzyMamdani.Defuzzifikasi(dataRule, hargaBatik)
        except:
            hasilDefuzz = 0

        if (status == 'train' or status == 'train50'):
            tempdatarule.append(np.nonzero(dataRule['MIN'].values)[0])

        prediksi.append(hasilDefuzz)
        dataMax.append(fuzzyMamdani.valueMax(dataRule))
    for i in range(len(prediksi)):
        dataPredAkt.append([aktual[i].astype(int), prediksi[i]])
    dataEvaluasi = fuzzyMamdani.MAPE(aktual, prediksi)

    if (status == 'train' or status == 'train50'):
        return fig, fig2, fuzzifikasi, dataMax, dataPredAkt, [dataEvaluasi], dataRule, tempdatarule
    else:
        return fig, fig2, fuzzifikasi, dataMax, dataPredAkt, [dataEvaluasi]


def mainGa():
    dataRule = pd.read_excel('assets/rule_data.xlsx')
    st.subheader('Pilih Partisi 🔽')
    optPar = st.selectbox('Partisi 75:25 atau 50:50',
                          ('[75:25]', '[50:50]'), index=0)
    if (optPar == '[75:25]'):
        status = "train"
    else:
        status = "train50"
    if (status == "train"):
        jk_Input = dataTrain['Harga Kain']
        lp_Input = dataTrain['Lama Pembuatan']
        m_Input = dataTrain['Motif']
        p_Input = dataTrain['Pewarnaan']
        aktual = dataTrain['Harga Aktual'].to_numpy()
    elif (status == "train50"):
        jk_Input = dataTrain50['Harga Kain']
        lp_Input = dataTrain50['Lama Pembuatan']
        m_Input = dataTrain50['Motif']
        p_Input = dataTrain50['Pewarnaan']
        aktual = dataTrain50['Harga Aktual'].to_numpy()
    semPem = [[13, 50], [1, 90], [10, 500], [3, 20], [60, 1800]]

    st.subheader('Set Parameter GA 🔽')
    optPop = st.selectbox('Populasi Size (10-200)', (20, 40, 60, 80, 100))
    optCr = st.selectbox('Crossover Rate (0-1)',
                         (0.2, 0.4, 0.6, 0.8, 1), index=0)
    optMr = st.selectbox('Mutation Rate (0-1)',
                         (0.2, 0.4, 0.6, 0.8, 1), index=0)
    optGen = st.selectbox('Generasi Size (10-200)', (20, 50, 80, 110, 140))

    if st.button('Tekan Untuk Running'):
        popTemp = []
        start_time = time.time()

        my_bar = st.progress(1, text="Please wait.")

        # konvergensi
        if (optGen == 30):
            desimal = 0.50
        elif (optGen == 50):
            desimal = 0.20
        else:
            desimal = 0.10

        tempcount = int(optGen * desimal)
        tempvalue = 1000

        for gene in range(optGen):
            tempdatarule = []
            if (gene == 0):
                pop = algoritmaGenetika.buat_populasi(optPop)
            else:
                pop = popTemp
            cross = algoritmaGenetika.crossover(pop, optCr)
            mut = algoritmaGenetika.mutation(cross, optMr)
            konv = algoritmaGenetika.konversi_kromosom(semPem, mut)
            tempFitness = []
            tempEval = []
            prediksi = []
            count = 0
            tempdatarule2 = []
            for i in range(len(konv)):
                Kain_murah = [13, 13, konv[i][0][1]]
                Kain_sedang = [konv[i][0][0],
                               (konv[i][0][0]+konv[i][0][3])/2, konv[i][0][3]]
                Kain_mahal = [konv[i][0][2], 52, 52]

                Lama_Pembuatan_Cepat = [1, 1, konv[i][1][1]]
                Lama_Pembuatan_Sedang = [
                    konv[i][1][0], (konv[i][1][0]+konv[i][1][3])/2, konv[i][1][3]]
                Lama_Pembuatan_Lama = [konv[i][1][2], 92, 92]

                Motif_Mudah = [10, 10, konv[i][2][1]]
                Motif_Sedang = [konv[i][2][0],
                                (konv[i][2][0]+konv[i][2][3])/2, konv[i][2][3]]
                Motif_Sulit = [konv[i][2][2], 502, 502]

                Pewarnaan_Murah = [3, 3, konv[i][3][1]]
                Pewarnaan_Sedang = [
                    konv[i][3][0], (konv[i][3][0]+konv[i][3][3])/2, konv[i][3][3]]
                Pewarnaan_Mahal = [konv[i][3][2], 22, 22]

                Batik_Murah = [60, 60, konv[i][4][1]]
                Batik_Sedang = [konv[i][4][0],
                                (konv[i][4][0]+konv[i][4][3])/2, konv[i][4][3]]
                Batik_Mahal = [konv[i][4][2], 1800, 1800]

                kain, lama, motif, pewarnaan, hargaBatik = fuzzyMamdani.memberFunction([Kain_murah, Kain_sedang, Kain_mahal], [Lama_Pembuatan_Cepat, Lama_Pembuatan_Sedang, Lama_Pembuatan_Lama], [
                                                                                       Motif_Mudah, Motif_Sedang, Motif_Sulit], [Pewarnaan_Murah, Pewarnaan_Sedang, Pewarnaan_Mahal], [Batik_Murah, Batik_Sedang, Batik_Mahal])
                for j in range(len(jk_Input)):
                    dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal = fuzzyMamdani.fuzzifikasi(
                        kain, lama, motif, pewarnaan, jk_Input[j], lp_Input[j], m_Input[j], p_Input[j])
                    fuzzyMamdani.Rule(dataRule, dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang,
                                      dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal)
                    try:
                        hasilDefuzz = fuzzyMamdani.Defuzzifikasi(
                            dataRule, hargaBatik)
                    except:
                        count += 1
                        hasilDefuzz = 1
                    tempdatarule2.append(np.nonzero(dataRule['MIN'].values)[0])
                    prediksi.append(hasilDefuzz)

                eval = fuzzyMamdani.MAPE(aktual, prediksi)
                tempEval.append(eval)
                prediksi = []
                fitn = algoritmaGenetika.fitnes(eval)
                tempFitness.append(fitn)
                tempdatarule.append([tempdatarule2])
                tempdatarule2=[]
            elit, index = algoritmaGenetika.elitism(
                np.array(tempEval), np.array(tempFitness), optPop)
            popTemp = []
            for i in range(len(elit)):
                popTemp.append(mut[index[i]])

            if (elit[0] < tempvalue):
                tempvalue = elit[0]
                tempcount = int(optGen * desimal)
            elif (tempvalue == elit[0]):
                if (tempcount != 0):
                    tempcount -= 1
            else:
                break

            if (gene == optGen-1):
                kondBut = True
                textBar = "Progress Finish"
            else:
                textBar = "Please wait."

            my_bar.progress(int((100/optGen)*(gene+1)), text=textBar)

        end_time = time.time()
        duration = end_time - start_time
        duration = duration // 60
        print(f"Lama komputasi: {duration} menit")

        individu = []
        dataMape = []
        fitness = []
        for i in range(10):
            col1 = konv[index[i]]
            col2 = elit[i]
            individu.append(col1)
            dataMape.append(col2)
            fitness.append(1/col2)

        model_terbaik = {
            'individu': individu,
            'mape': dataMape,
            'fitness': fitness
        }

        status_data = {
            'status': [status],
        }

        df_model_terbaik = pd.DataFrame(model_terbaik)
        file_name = 'model_best.xlsx'

        algoritmaGenetika.EliminasiRuleGA(dataRule, tempdatarule[index[0]][0])

        df_status = pd.DataFrame(status_data)
        file_name2 = 'status.xlsx'

        if os.path.exists(file_name):
            os.remove(file_name)

        if os.path.exists(file_name2):
            os.remove(file_name2)

        df_model_terbaik.to_excel(file_name, index=False)
        df_status.to_excel(file_name2, index=False)


with st.container():
    if (selected == 'Home'):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**OPTIMASI FUNGSI KEANGGOTAAN FUZZY MAMDANI DENGAN ALGORITMA GENETIKA PADA PENENTUAN HARGA PRODUKSI BATIK (STUDI KASUS : BANGKALAN MADURA)**")
            st.markdown("Nama : Muhammad Fahmi Ady Susilo")
            st.markdown("Nim : 190411100127")
            st.markdown("Bidang : Kecerdasan Komputasional")
            st.markdown("Prodi : Teknik Informatika")
        with col2:
            st.image("assets/batik.jpg")

    elif (selected == 'Dataset'):
        st.header('Dataset 🔽')
        st.dataframe(data, use_container_width=True, hide_index=True)
        tab1, tab2 = st.tabs(["Data 75:25", "Data 50:50"])
        with tab1:
            st.header('Data Training 75% 🔽')
            st.dataframe(dataTrain, use_container_width=True, hide_index=True)
            st.header('Data Testing 25% 🔽')
            st.dataframe(dataTest, use_container_width=True, hide_index=True)
        with tab2:
            st.header('Data Training 50% 🔽')
            st.dataframe(dataTrain50, use_container_width=True,
                         hide_index=True)
            st.header('Data Testing 50% 🔽')
            st.dataframe(dataTest50, use_container_width=True, hide_index=True)
    elif (selected == 'Rule Aturan'):
        st.header('Rule Aturan Fuzzy 🔽')
        st.dataframe(dataRule, use_container_width=True, hide_index=True)

    elif (selected == 'Fuzzy Mamdani'):
        st.header('Fuzzy Mamdani')

        fig, fig2, fuzzifikasi, dataMax, dataPredAkt, dataEvaluasi, dataRule1, tempdatarule1 = mainFuzzy(
            'train')
        fig, fig2, fuzzifikasi2, dataMax2, dataPredAkt2, dataEvaluasi2 = mainFuzzy(
            'test')
        fig, fig2, fuzzifikasi3, dataMax3, dataPredAkt3, dataEvaluasi3, dataRule3, tempdatarule3 = mainFuzzy(
            'train50')
        fig, fig2, fuzzifikasi4, dataMax4, dataPredAkt4, dataEvaluasi4 = mainFuzzy(
            'test50')

        fuzzifikasi = pd.DataFrame(fuzzifikasi, columns=['Harga Kain Murah', 'Harga Kain Sedang', 'Harga Kain Mahal', 'Lama Pembuatan Cepat', 'Lama Pembuatan Sedang',
                                   'Lama Pembuatan Lama', 'Motif Mudah', 'Motif Sedang', 'Motif Sulit', 'Pewarnaan  Murah', 'Pewarnaan  Sedang', 'Pewarnaan  Mahal'])
        fuzzifikasi2 = pd.DataFrame(fuzzifikasi2, columns=['Harga Kain Murah', 'Harga Kain Sedang', 'Harga Kain Mahal', 'Lama Pembuatan Cepat', 'Lama Pembuatan Sedang',
                                    'Lama Pembuatan Lama', 'Motif Mudah', 'Motif Sedang', 'Motif Sulit', 'Pewarnaan  Murah', 'Pewarnaan  Sedang', 'Pewarnaan  Mahal'])
        fuzzifikasi3 = pd.DataFrame(fuzzifikasi3, columns=['Harga Kain Murah', 'Harga Kain Sedang', 'Harga Kain Mahal', 'Lama Pembuatan Cepat', 'Lama Pembuatan Sedang',
                                                           'Lama Pembuatan Lama', 'Motif Mudah', 'Motif Sedang', 'Motif Sulit', 'Pewarnaan  Murah', 'Pewarnaan  Sedang', 'Pewarnaan  Mahal'])
        fuzzifikasi4 = pd.DataFrame(fuzzifikasi4, columns=['Harga Kain Murah', 'Harga Kain Sedang', 'Harga Kain Mahal', 'Lama Pembuatan Cepat', 'Lama Pembuatan Sedang',
                                    'Lama Pembuatan Lama', 'Motif Mudah', 'Motif Sedang', 'Motif Sulit', 'Pewarnaan  Murah', 'Pewarnaan  Sedang', 'Pewarnaan  Mahal'])

        dataMax = pd.DataFrame(dataMax, columns=['Murah', 'Sedang', 'Mahal'])
        dataMax2 = pd.DataFrame(dataMax2, columns=['Murah', 'Sedang', 'Mahal'])
        dataMax3 = pd.DataFrame(dataMax3, columns=['Murah', 'Sedang', 'Mahal'])
        dataMax4 = pd.DataFrame(dataMax4, columns=['Murah', 'Sedang', 'Mahal'])

        dataPredAkt = pd.DataFrame(dataPredAkt, columns=['Aktual', 'Prediksi'])
        dataPredAkt2 = pd.DataFrame(
            dataPredAkt2, columns=['Aktual', 'Prediksi'])
        dataPredAkt3 = pd.DataFrame(
            dataPredAkt3, columns=['Aktual', 'Prediksi'])
        dataPredAkt4 = pd.DataFrame(
            dataPredAkt4, columns=['Aktual', 'Prediksi'])

        dataEvaluasi = pd.DataFrame(dataEvaluasi, columns=['MAPE'])
        dataEvaluasi2 = pd.DataFrame(dataEvaluasi2, columns=['MAPE'])
        dataEvaluasi3 = pd.DataFrame(dataEvaluasi3, columns=['MAPE'])
        dataEvaluasi4 = pd.DataFrame(dataEvaluasi4, columns=['MAPE'])

        st.subheader('Member Function 🔽')
        st.text('Representasi Kurva Member Function')
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig)
        with col2:
            st.pyplot(fig2)

        st.subheader('Pilih Partisi 🔽')
        par1, par2 = st.tabs(["Partisi 75:25", "Partisi 50:50"])
        with par1:
            st.subheader('Pilih Proses 🔽')
            tab1, tab2 = st.tabs(["Training", "Testing"])
            with tab1:
                st.subheader('Fuzzifikasi 🔽')
                st.text('Hasil proses Fuzzifikasi')
                fuzzifikasi.reset_index(drop=True, inplace=True)
                fuzzifikasi.index = fuzzifikasi.index + 1
                fuzzifikasi = fuzzifikasi.rename_axis('Dataset')
                st.dataframe(fuzzifikasi, use_container_width=True)
                st.subheader('Implikasi MIN 🔽')
                st.text('Data terlalu banyak untuk ditampilkan')
                st.success('Selesai', icon="✅")
                st.subheader('Agregasi MAX 🔽')
                st.text('Hasil proses Agregasi MAX')
                dataMax.reset_index(drop=True, inplace=True)
                dataMax.index = dataMax.index + 1
                dataMax = dataMax.rename_axis('Dataset')
                st.dataframe(dataMax, use_container_width=True)
                st.subheader('Defuzzifikasi 🔽')
                st.text('Hasil proses Defuzzifikasi Centroid')

                dataPredAkt.reset_index(drop=True, inplace=True)
                dataPredAkt.index = dataPredAkt.index + 1
                dataPredAkt = dataPredAkt.rename_axis('Dataset')
                st.dataframe(dataPredAkt, use_container_width=True)
                chart_data = pd.DataFrame(dataPredAkt)
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )

                # dataDefuzz = pd.read_excel(
                #     'D:/Code_skripsi/streamlit/assets/result.xlsx', sheet_name='sheet2')

                # dataDefuzz.reset_index(drop=True, inplace=True)
                # dataDefuzz.index = dataDefuzz.index + 1
                # dataDefuzz = dataDefuzz.rename_axis('Dataset')
                # st.dataframe(dataDefuzz, use_container_width=True)
                # chart_data = pd.DataFrame(dataDefuzz)
                # st.line_chart(
                #     chart_data,
                #     use_container_width=True
                # )

                st.subheader('Evaluasi 🔽')
                st.text('Hasil proses Evaluasi MAPE')
                st.dataframe(dataEvaluasi, use_container_width=True,
                             hide_index=True)
                st.subheader('Model Rule Based Terpilih 🔽')
                fuzzyMamdani.EliminasiRule(dataRule1, tempdatarule1)
                rule = pd.read_excel('modelRule_best.xlsx')
                st.dataframe(rule, use_container_width=True,
                             hide_index=True)
                st.text(f'Total Dataset : {len(rule)}')
            with tab2:
                st.subheader('Fuzzifikasi 🔽')
                st.text('Hasil proses Fuzzifikasi')
                fuzzifikasi2.reset_index(drop=True, inplace=True)
                fuzzifikasi2.index = fuzzifikasi2.index + 1
                fuzzifikasi2 = fuzzifikasi2.rename_axis('Dataset')
                st.dataframe(fuzzifikasi2, use_container_width=True)
                st.subheader('Implikasi MIN 🔽')
                st.text('Data terlalu banyak untuk ditampilkan')
                st.success('Selesai', icon="✅")
                st.subheader('Agregasi MAX 🔽')
                st.text('Hasil proses Agregasi MAX')
                dataMax2.reset_index(drop=True, inplace=True)
                dataMax2.index = dataMax2.index + 1
                dataMax2 = dataMax2.rename_axis('Dataset')
                st.dataframe(dataMax2, use_container_width=True)
                st.subheader('Defuzzifikasi 🔽')
                st.text('Hasil proses Defuzzifikasi Centroid')
                dataPredAkt2.reset_index(drop=True, inplace=True)
                dataPredAkt2.index = dataPredAkt2.index + 1
                dataPredAkt2 = dataPredAkt2.rename_axis('Dataset')
                st.dataframe(dataPredAkt2, use_container_width=True)
                chart_data = pd.DataFrame(dataPredAkt2)
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )
                st.subheader('Evaluasi 🔽')
                st.text('Hasil proses Evaluasi MAPE')
                st.dataframe(dataEvaluasi2, use_container_width=True,
                             hide_index=True)
        with par2:
            st.subheader('Pilih Proses 🔽')
            tab1, tab2 = st.tabs(["Training", "Testing"])
            with tab1:
                st.subheader('Fuzzifikasi 🔽')
                st.text('Hasil proses Fuzzifikasi')
                fuzzifikasi3.reset_index(drop=True, inplace=True)
                fuzzifikasi3.index = fuzzifikasi3.index + 1
                fuzzifikasi3 = fuzzifikasi3.rename_axis('Dataset')
                st.dataframe(fuzzifikasi3, use_container_width=True)
                st.subheader('Implikasi MIN 🔽')
                st.text('Data terlalu banyak untuk ditampilkan')
                st.success('Selesai', icon="✅")
                st.subheader('Agregasi MAX 🔽')
                st.text('Hasil proses Agregasi MAX')
                dataMax3.reset_index(drop=True, inplace=True)
                dataMax3.index = dataMax3.index + 1
                dataMax3 = dataMax3.rename_axis('Dataset')
                st.dataframe(dataMax3, use_container_width=True)
                st.subheader('Defuzzifikasi 🔽')
                st.text('Hasil proses Defuzzifikasi Centroid')
                dataPredAkt3.reset_index(drop=True, inplace=True)
                dataPredAkt3.index = dataPredAkt3.index + 1
                dataPredAkt3 = dataPredAkt3.rename_axis('Dataset')
                st.dataframe(dataPredAkt3, use_container_width=True)
                chart_data = pd.DataFrame(dataPredAkt3)
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )
                st.subheader('Evaluasi 🔽')
                st.text('Hasil proses Evaluasi MAPE')
                st.dataframe(dataEvaluasi3, use_container_width=True,
                             hide_index=True)
                st.subheader('Model Rule Based Terpilih 🔽')
                fuzzyMamdani.EliminasiRule(dataRule3, tempdatarule3)
                rule3 = pd.read_excel('modelRule_best.xlsx')
                st.dataframe(rule3, use_container_width=True,
                             hide_index=True)

            with tab2:
                st.subheader('Fuzzifikasi 🔽')
                st.text('Hasil proses Fuzzifikasi')
                fuzzifikasi4.reset_index(drop=True, inplace=True)
                fuzzifikasi4.index = fuzzifikasi4.index + 1
                fuzzifikasi4 = fuzzifikasi4.rename_axis('Dataset')
                st.dataframe(fuzzifikasi4, use_container_width=True)
                st.subheader('Implikasi MIN 🔽')
                st.text('Data terlalu banyak untuk ditampilkan')
                st.success('Selesai', icon="✅")
                st.subheader('Agregasi MAX 🔽')
                st.text('Hasil proses Agregasi MAX')
                dataMax4.reset_index(drop=True, inplace=True)
                dataMax4.index = dataMax4.index + 1
                dataMax4 = dataMax4.rename_axis('Dataset')
                st.dataframe(dataMax4, use_container_width=True)
                st.subheader('Defuzzifikasi 🔽')
                st.text('Hasil proses Defuzzifikasi Centroid')
                dataPredAkt4.reset_index(drop=True, inplace=True)
                dataPredAkt4.index = dataPredAkt4.index + 1
                dataPredAkt4 = dataPredAkt4.rename_axis('Dataset')
                st.dataframe(dataPredAkt4, use_container_width=True)
                chart_data = pd.DataFrame(dataPredAkt4)
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )
                st.subheader('Evaluasi 🔽')
                st.text('Hasil proses Evaluasi MAPE')
                st.dataframe(dataEvaluasi4, use_container_width=True,
                             hide_index=True)
    else:
        dataPredAkt3 = []
        st.header('Fuzzy Mamdani & Algoritma Genetika')
        tab1, tab2 = st.tabs(["Optimasi Fuzzy", "Implementasi"])
        with tab1:
            pil1, pil2 = st.tabs(["Data Training", "Data Testing"])
            with pil1:
                mainGa()
                model_best = pd.read_excel('model_best.xlsx')
                st.dataframe(model_best, use_container_width=True,
                             hide_index=True)

                st.subheader('Rule Based Terpilih 🔽')
                modelRuleGA_best = pd.read_excel('modelRuleGA_best.xlsx')
                st.dataframe(modelRuleGA_best, use_container_width=True,
                             hide_index=True)
                st.text(f'Total Dataset : {len(modelRuleGA_best)}')

                st.subheader('Model Terbaik 🔽')
                st.dataframe(model_best.head(
                    1), use_container_width=True, hide_index=True)
            with pil2:
                st.subheader('Solusi Terbaik 🔽')
                model = pd.read_excel('model_best.xlsx')
                status = pd.read_excel('status.xlsx')
                model_best = ast.literal_eval(model['individu'][0])
                st.text(model_best)
                modelRuleGA_best = pd.read_excel('modelRuleGA_best.xlsx')
                # dataRule=modelRuleGA_best
                prediksi = []
                count = 0

                if (status['status'][0] == "train"):
                    jk_Input = dataTest['Harga Kain']
                    lp_Input = dataTest['Lama Pembuatan']
                    m_Input = dataTest['Motif']
                    p_Input = dataTest['Pewarnaan']
                    aktual = dataTest['Harga Aktual'].to_numpy()
                else:
                    jk_Input = dataTest50['Harga Kain']
                    lp_Input = dataTest50['Lama Pembuatan']
                    m_Input = dataTest50['Motif']
                    p_Input = dataTest50['Pewarnaan']
                    aktual = dataTest50['Harga Aktual'].to_numpy()

                Kain_murah = [13, 13, model_best[0][1]]
                Kain_sedang = [
                    model_best[0][0], (model_best[0][0] + model_best[0][3]) / 2, model_best[0][3]]
                Kain_mahal = [model_best[0][2], 52, 52]

                Lama_Pembuatan_Cepat = [1, 1, model_best[1][1]]
                Lama_Pembuatan_Sedang = [
                    model_best[1][0], (model_best[1][0] + model_best[1][3]) / 2, model_best[1][3]]
                Lama_Pembuatan_Lama = [model_best[1][2], 92, 92]

                Motif_Mudah = [10, 10, model_best[2][1]]
                Motif_Sedang = [
                    model_best[2][0], (model_best[2][0] + model_best[2][3]) / 2, model_best[2][3]]
                Motif_Sulit = [model_best[2][2], 502, 502]

                Pewarnaan_Murah = [3, 3, model_best[3][1]]
                Pewarnaan_Sedang = [
                    model_best[3][0], (model_best[3][0] + model_best[3][3]) / 2, model_best[3][3]]
                Pewarnaan_Mahal = [model_best[3][2], 22, 22]

                Batik_Murah = [60, 60, model_best[4][1]]
                Batik_Sedang = [
                    model_best[4][0], (model_best[4][0] + model_best[4][3]) / 2, model_best[4][3]]
                Batik_Mahal = [model_best[4][2], 1800, 1800]

                kain, lama, motif, pewarnaan, hargaBatik = fuzzyMamdani.memberFunction([Kain_murah, Kain_sedang, Kain_mahal], [Lama_Pembuatan_Cepat, Lama_Pembuatan_Sedang, Lama_Pembuatan_Lama], [
                                                                                       Motif_Mudah, Motif_Sedang, Motif_Sulit], [Pewarnaan_Murah, Pewarnaan_Sedang, Pewarnaan_Mahal], [Batik_Murah, Batik_Sedang, Batik_Mahal])
                for j in range(len(jk_Input)):
                    dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal = fuzzyMamdani.fuzzifikasi(
                        kain, lama, motif, pewarnaan, jk_Input[j], lp_Input[j], m_Input[j], p_Input[j])
                    fuzzyMamdani.Rule(dataRule, dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang,
                                      dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal)
                    try:
                        hasilDefuzz = fuzzyMamdani.Defuzzifikasi(
                            dataRule, hargaBatik)
                    except:
                        count += 1
                        hasilDefuzz = 1
                    prediksi.append(hasilDefuzz)
                eval = fuzzyMamdani.MAPE(aktual, prediksi)

                for i in range(len(prediksi)):
                    dataPredAkt3.append([aktual[i].astype(int), prediksi[i]])

                dataEvaluasi = fuzzyMamdani.MAPE(aktual, prediksi)

                dataPredAkt3 = pd.DataFrame(
                    dataPredAkt3, columns=['Aktual', 'Prediksi'])
                dataEvaluasi = pd.DataFrame([dataEvaluasi], columns=['MAPE'])

                st.subheader('Defuzzifikasi 🔽')
                st.text('Hasil proses Defuzzifikasi Centroid')
                st.dataframe(
                    dataPredAkt3, use_container_width=True, hide_index=True)
                chart_data = pd.DataFrame(dataPredAkt3)
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )
                st.subheader('Evaluasi 🔽')
                st.text('Hasil proses Evaluasi MAPE')
                st.dataframe(
                    dataEvaluasi, use_container_width=True, hide_index=True)
        with tab2:
            model = pd.read_excel('model_best.xlsx')
            model_best = ast.literal_eval(model['individu'][0])
            st.subheader('Implementasi 🔽')
            st.text('Masukkan Data pada column yang disediakan')
            jk_InputT = st.selectbox('Masukkan Input Harga Kain', (22, 34, 44))
            lp_InputT = st.number_input(
                "Masukkan Input Lama Pembuatan (1-90) :", min_value=1, max_value=90, value=1, step=1)
            m_InputT = st.number_input(
                "Masukkan Input Motif (10-500) :", min_value=10, max_value=500, value=10, step=1)
            p_InputT = st.selectbox(
                'Masukkan Input Pewarnaan', (3, 6, 9, 12, 16, 20))
            st.subheader('Hasil 🔽')
            if (jk_InputT != None and lp_InputT != None and m_InputT != None and p_InputT != None):
                Kain_murah = [13, 13, model_best[0][1]]
                Kain_sedang = [
                    model_best[0][0], (model_best[0][0] + model_best[0][3]) / 2, model_best[0][3]]
                Kain_mahal = [model_best[0][2], 52, 52]

                Lama_Pembuatan_Cepat = [1, 1, model_best[1][1]]
                Lama_Pembuatan_Sedang = [
                    model_best[1][0], (model_best[1][0] + model_best[1][3]) / 2, model_best[1][3]]
                Lama_Pembuatan_Lama = [model_best[1][2], 92, 92]

                Motif_Mudah = [10, 10, model_best[2][1]]
                Motif_Sedang = [
                    model_best[2][0], (model_best[2][0] + model_best[2][3]) / 2, model_best[2][3]]
                Motif_Sulit = [model_best[2][2], 502, 502]

                Pewarnaan_Murah = [3, 3, model_best[3][1]]
                Pewarnaan_Sedang = [
                    model_best[3][0], (model_best[3][0] + model_best[3][3]) / 2, model_best[3][3]]
                Pewarnaan_Mahal = [model_best[3][2], 22, 22]

                Batik_Murah = [60, 60, model_best[4][1]]
                Batik_Sedang = [
                    model_best[4][0], (model_best[4][0] + model_best[4][3]) / 2, model_best[4][3]]
                Batik_Mahal = [model_best[4][2], 1800, 1800]

                kain, lama, motif, pewarnaan, hargaBatik = fuzzyMamdani.memberFunction([Kain_murah, Kain_sedang, Kain_mahal], [Lama_Pembuatan_Cepat, Lama_Pembuatan_Sedang, Lama_Pembuatan_Lama], [
                                                                                       Motif_Mudah, Motif_Sedang, Motif_Sulit], [Pewarnaan_Murah, Pewarnaan_Sedang, Pewarnaan_Mahal], [Batik_Murah, Batik_Sedang, Batik_Mahal])
                dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang, dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal = fuzzyMamdani.fuzzifikasi(
                    kain, lama, motif, pewarnaan, jk_InputT, lp_InputT, m_InputT, p_InputT)
                fuzzyMamdani.Rule(dataRule, dr_kain_murah, dr_kain_sedang, dr_kain_mahal, dr_LP_cepat, dr_LP_sedang,
                                  dr_LP_lama, dr_M_mudah, dr_M_sedang, dr_M_sulit, dr_P_murah, dr_P_sedang, dr_P_mahal)
                try:
                    hasilDefuzz = fuzzyMamdani.Defuzzifikasi(
                        dataRule, hargaBatik)
                except:
                    hasilDefuzz = 1
                st.write("<p>Harga Jual kain : <span style='color:green'>{}</span></p>".format(
                    hasilDefuzz), unsafe_allow_html=True)
            else:
                st.write(
                    "<p>Harga Jual kain : <span style='color:green'>0</span></p>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
