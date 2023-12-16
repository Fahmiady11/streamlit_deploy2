import random
import numpy as np
import pandas as pd
import os

# Semesta Pembicaraan
J_Kain = np.arange(13, 52, 1)
L_Pembuatan = np.arange(1, 92, 1)
M_otif = np.arange(10, 502, 1)
P_ewarnaan = np.arange(3, 22, 1)
H_Batik = np.arange(60, 1800, 1)


def buat_populasi(ukuran):
    new_pop_utama = []
    new_pop = []
    for pop in range(ukuran):
        for i in range(20):
            rand = random.randint(0, 1000)
            new_pop.append(rand)
        new_pop_utama.append(new_pop)
        new_pop = []
    return new_pop_utama

# Crossover


def crossover(parent, pr):
    total_child = len(parent)*pr
    total_child = round(total_child)
    child = []
    while (len(child) < total_child):
        rand1 = random.randint(0, len(parent)-1)
        rand2 = random.randint(0, len(parent)-1)
        if (rand1 != rand2):
            flattened_list1 = parent[rand1]
            flattened_list2 = parent[rand2]
            # proses silang
            pointRand1 = random.randint(0, len(flattened_list1)-1)
            pointRand2 = random.randint(0, len(flattened_list1)-1)
            pointMin = min(pointRand1, pointRand2)
            pointMax = max(pointRand1, pointRand2)
            slicePop1 = flattened_list1[pointMin:pointMax+1]
            slicePop2 = flattened_list2[pointMin:pointMax+1]
            hasilSilang1 = flattened_list1[:pointMin] + \
                slicePop2 + flattened_list1[pointMax+1:]
            hasilSilang2 = flattened_list2[:pointMin] + \
                slicePop1 + flattened_list2[pointMax+1:]
            if (len(child)+1 < total_child):
                child.append(hasilSilang1)
                child.append(hasilSilang2)
            else:
                child.append(hasilSilang1)
    for i in range(len(parent)):
        child.insert(i, parent[i])
    return child

# Mutasi


def mutation(parent, pm):
    total_child = len(parent)*pm
    total_child = round(total_child)
    total_rand = len(parent[0])*pm
    total_rand = round(total_rand)
    child = []
    temp = []
    for i in range(total_child):
        rand = random.randint(0, len(parent)-1)
        randMutation = random.randint(0, len(parent)-1)
        parentRand = parent[rand]
        for j in range(total_rand):
            cond = True
            while (cond):
                randMut = random.randint(0, len(parent[i])-1)
                genRand = parentRand[randMut]
                if (randMut not in temp):
                    temp.append(randMut)
                    randVal = random.uniform(-0.1, 0.1)
                    mutation = abs(
                        round(min(random.randint(900, 1000), genRand+(randVal*(1000-1)))))
                    parentRand = parentRand[:randMut] + \
                        [mutation]+parentRand[randMut+1:]
                    cond = False
        temp = []
        child.append(parentRand)
    for i in range(len(parent)):
        child.insert(i, parent[i])
    return child

# Konversi Kromosom


def konversi_kromosom(sem, data):
    dataSort = []
    temp = []
    temp1 = []
    for i in range(len(data)):
        temp4 = []
        count4 = 0
        for j in range(len(data[i])):
            temp4.append(data[i][j])
            count4 += 1
            if (count4 == 4):
                temp1.append(temp4)
                count4 = 0
                temp4 = []
        temp.append(temp1)
        temp1 = []
    l = 0
    for k in range(len(temp)):  # 10
        for l in range(len(temp[k])):  # kromosom 4
            for m in range(len(temp[k][l])):  # gen
                konversi = round(
                    ((temp[k][l][m]/1000)*(sem[l][1]-sem[l][0]))+sem[l][0])
                temp[k][l][m] = konversi
                # print(k,l,m,temp[k][l][m],konversi,sem[l][0],sem[l][1])
        sorted_data = [sorted(sublist) for sublist in temp[k]]
        dataSort.append(sorted_data)
    return dataSort


def fitnes(mape):
    try:
        output = 1 / mape
    except ZeroDivisionError:
        output = 0
    return output


def elitism(population, fitness, elite_size):
    sort_index = np.argsort(fitness)
    sorted_population = population[sort_index]
    elite = sorted_population[-elite_size:]
    return elite[::-1].flatten(), sort_index[::-1].flatten()


def EliminasiRuleGA(dataRule, tempdatarule, index):
    tempdatarule = np.concatenate(tempdatarule[index[0]][0])
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
    file_name = 'modelRuleGA_best.xlsx'
    if os.path.exists(file_name):
        os.remove(file_name)
    df_rule_terbaik.to_excel(file_name, index=False)