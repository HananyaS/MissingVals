import os
import pandas as pd
import numpy as np
import random

features_list = []

random.seed(10)


def is_nan(x):
    return (x != x)


list_to_remove = []

annotations = pd.read_excel('Data/SapirsData/raw/efsdict_annotated_2021-04-21.xlsx')

"""for row in annotations.iterrows():
    print(row[1]["field available at prelim search"])
    if row[1]["field available at prelim search"] == 0.0:
        list_to_remove.append(row[1]["Variable"])"""
list_to_remove = [row[1]["Variable"] for row in annotations.iterrows() if
                  row[1]["field available at prelim search"] == 0]
list_importent = ["venthxpr", "funghxpr", "condclas", "gvhprhrxgp", "priauto", "coorgscore", "hctcigpb", "invivotcd"
                                                                                                         "karnofcat",
                  "karnofraw", "tbigp", "hxmalig"]  # "abotyper"
list_importent_only_AML = ["mdstfaml", "amlrxrel", "indcycle", "cytogeneeln", "aldist"]
# list_importent += list_importent_only_AML
list_to_remove += list_importent_only_AML
input_data_efs = pd.read_excel('Data/SapirsData/raw/pubefsdatadp_shuffled.xlsx')
print(len(input_data_efs))

input_data_efs = input_data_efs.drop(
    ["abotyper", 'drsex', 'dpb1nonpermissive', 'tag_consent', 'tag_noembargo', 'indxtxgp', 'rid_rollup_race',
     'did_rollup_race', 'income_gp', 'pseudoccn', 'multicord', 'multprod', 'dnrgrp1', 'drabomatch', 'dabotype',
     'ethnicit', 'r_drb3_typ1', 'r_drb3_typ2', 'r_drb4_typ1', 'r_drb4_typ2', 'r_drb5_typ1', 'r_drb5_typ2',
     'r_dqa1_typ1', 'r_dqa1_typ2', 'r_dpa1_typ1', 'r_dpa1_typ2', 'r_dpb1_typ1', 'r_dpb1_typ2'],
    axis=1)  # 'prior_auto', 'brd_dis',
input_data_efs = input_data_efs.drop(
    ['numhires12_1', 'permiss3', 'himatchdpb1_1', 'Unnamed: 0', 'd_drb3_typ1', 'd_drb3_typ2', 'd_drb4_typ1',
     'd_drb4_typ2', 'd_drb5_typ1', 'd_drb5_typ2', 'd_dqa1_typ1', 'd_dqa1_typ2', 'd_dpa1_typ1', 'd_dpa1_typ2',
     'd_dpb1_typ1', 'd_dpb1_typ2', 'related', 'dnrtype', 'urdbmpbhlagp', 'donorgpnew'], axis=1)
input_data_efs = input_data_efs.drop(['urdbmpbdeth', 'urdbmpbdagegp', 'agegp', 'rcmvpr', 'yeartx', 'ccn'], axis=1)

input_data_efs.loc[input_data_efs.hxmalig == 99, 'hxmalig'] = 0

input_data_efs.loc[input_data_efs.racegp == 3, 'racegp'] = 2
input_data_efs.loc[input_data_efs.racegp == 4, 'racegp'] = 2
input_data_efs.loc[input_data_efs.racegp == 5, 'racegp'] = 2
input_data_efs.loc[input_data_efs.racegp == 7, 'racegp'] = 2
input_data_efs.loc[input_data_efs.racegp == 8, 'racegp'] = 2

input_data_efs.loc[input_data_efs.urdbmpbdrace == 3, 'urdbmpbdrace'] = 2
input_data_efs.loc[input_data_efs.urdbmpbdrace == 4, 'urdbmpbdrace'] = 2
input_data_efs.loc[input_data_efs.urdbmpbdrace == 5, 'urdbmpbdrace'] = 2
input_data_efs.loc[input_data_efs.urdbmpbdrace == 7, 'urdbmpbdrace'] = 2
input_data_efs.loc[input_data_efs.urdbmpbdrace == 8, 'urdbmpbdrace'] = 2

# dropped after feature importance
# input_data_efs = input_data_efs.drop(['priauto', 'disease', 'gvhprhrxgp'], axis=1)
list_y = []

list_class = [('crel', 'intxcrel'), ('mscgvhd', 'intxmscgvhd'), ('dead', 'intxsurv'), ('rej', 'intxrej'),
              ('efs', 'intxefs')]
for label in list_class:
    count = 0
    count2 = 0
    column_event = label[0]
    column_intx = label[1]

    input_data_efs[column_event] = input_data_efs[column_event].apply(lambda x: x if (x == 0 or x == 1) else np.nan)

    for i in range(len(input_data_efs)):
        intx = input_data_efs.loc[i, column_intx]
        event = input_data_efs.loc[i, column_event]
        if is_nan(intx) == True or is_nan(event) == True:
            input_data_efs.loc[i, column_event] = np.nan
            input_data_efs.loc[i, column_intx] = np.nan
        else:
            if event == 0 and intx >= 12:
                input_data_efs.loc[i, column_event] = 0
            elif event == 1 and intx < 12:
                input_data_efs.loc[i, column_event] = 1
            elif event == 1 and intx >= 12:
                input_data_efs.loc[i, column_event] = 0
            else:
                input_data_efs.loc[i, column_event] = np.nan

            if event == 0 and intx >= 24:
                input_data_efs.loc[i, column_intx] = 0
            elif event == 1 and intx < 24:
                input_data_efs.loc[i, column_intx] = 1
            elif event == 1 and intx >= 24:
                input_data_efs.loc[i, column_intx] = 0
            else:
                input_data_efs.loc[i, column_intx] = np.nan

    print(label)
d = {'rej': 'rej1y', 'intxrej': 'rej2y', 'crel': 'rel1y', 'intxcrel': 'rel2y', 'mscgvhd': 'gvhd1y',
     'intxmscgvhd': 'gvhd2y', 'dead': 'dead1y', 'intxsurv': 'dead2y', 'efs': 'efs1y', 'intxefs': 'efs2y'}
input_data_efs = input_data_efs.rename(columns=d)
list_y = list(d.values())

# remove dead after 1y but no occurane appened
list_1y = ['rel1y', 'rej1y', 'gvhd1y']
"""for label in list_1y:
    data_efs[label] = data_efs[label].apply(lambda x: np.nan if  (x == 0 and data_efs['dead1y'] == 1) else x)"""
list_2y = ['rel2y', 'rej2y', 'gvhd2y']
for i in range(len(input_data_efs)):
    for label in list_1y:
        if input_data_efs.loc[i, 'dead1y'] == 1 and input_data_efs.loc[i, label] == 0:
            input_data_efs.loc[i, label] = np.nan
    for label in list_2y:
        if input_data_efs.loc[i, 'dead2y'] == 1 and input_data_efs.loc[i, label] == 0:
            input_data_efs.loc[i, label] = np.nan

"""dict_dist_class1and2 = {}
with open('../calc_dist_from_ml_paper/aa_class1.fas_PairwiseDistanceList.txt') as dist_f:
    for line in dist_f:
        if line != 'Allele_pairs	Distance\n':
            line = line.strip().split('\t')
            dict_dist_class1and2[line[0]] = float(line[1])

dist_f.close()"""

"""dict_pseudoid_muugs = {}
with open('efs.umug.freqs') as muugs_freqs:
    for line in muugs_freqs:
        line = line.strip().split(',')
        if not line[0] in dict_pseudoid_muugs:
            dict_pseudoid_muugs[line[0]] = line[1]
muugs_freqs.close()"""

"""column_allels = ['_a_typ1','_a_typ2','_b_typ1','_b_typ2','_c_typ1','_c_typ2', '_dqb1_typ1','_dqb1_typ2', '_drb1_typ1','_drb1_typ2']
for i in range(len(input_data_efs)):
    id_ = str(input_data_efs.loc[i,'pseudoid'])
    id_r = id_ + 'r'
    if id_r in dict_pseudoid_muugs:
        muug_r = dict_pseudoid_muugs[id_ +'r'].replace('+', '^').split('^')
        muug_d = dict_pseudoid_muugs[id_  + 'd'].replace('+', '^').split('^')
        for allele_idx in range(len(column_allels)):
            input_data_efs.loc[i, 'r' + column_allels[allele_idx]] = muug_r[allele_idx]
            input_data_efs.loc[i, 'd' + column_allels[allele_idx]] = muug_d[allele_idx]
#input_data_efs = input_data_efs.drop(['studyid'],axis=1)"""
print('jj')
list_allels_count = []
list_allels = [['r_a_typ1', 'r_a_typ2', 'd_a_typ1', 'd_a_typ2'], ['r_b_typ1', 'r_b_typ2', 'd_b_typ1', 'd_b_typ2'],
               ['r_c_typ1', 'r_c_typ2', 'd_c_typ1', 'd_c_typ2'],
               ['r_drb1_typ1', 'r_drb1_typ2', 'd_drb1_typ1', 'd_drb1_typ2'],
               ['r_dqb1_typ1', 'r_dqb1_typ2', 'd_dqb1_typ1', 'd_dqb1_typ2']]

for j, allels in enumerate(list_allels):
    indexNamesArr = input_data_efs.index.values
    for i in indexNamesArr:
        count_similar = 0
        if not pd.isnull(input_data_efs.loc[i, allels[0]]):
            if input_data_efs.loc[i, allels[0]] == input_data_efs.loc[i, allels[2]]:
                if input_data_efs.loc[i, allels[1]] == input_data_efs.loc[i, allels[3]]:
                    count_similar += 2
                else:
                    count_similar += 1
            elif input_data_efs.loc[i, allels[1]] == input_data_efs.loc[i, allels[3]]:
                count_similar += 1
            elif input_data_efs.loc[i, allels[0]] == input_data_efs.loc[i, allels[3]]:
                count_similar += 1
            elif input_data_efs.loc[i, allels[1]] == input_data_efs.loc[i, allels[2]]:
                count_similar += 1
            if j < 3:
                a1 = input_data_efs.loc[i, allels[0]].replace('*', '').replace(':', '')
                a2 = input_data_efs.loc[i, allels[1]].replace('*', '').replace(':', '')
                a = (' _ ').join(sorted([a1, a2]))
                k = 0
                dist_alleles = np.nan

                # Added comment
                """
                if a in dict_dist_class1and2:
                    dist_alleles = dict_dist_class1and2[a]
                elif a1 == a2:
                    dist_alleles = 0
                """

                input_data_efs.loc[i, allels[1]] = dist_alleles
            input_data_efs.loc[i, allels[0]] = count_similar + 1

    list_allels_count.append(allels[0])
    if j < 3:
        list_allels_count.append(allels[1])
        del allels[1]
    del allels[0]
    input_data_efs = input_data_efs.drop(allels, axis=1)

###check impute res
count_unmatch = 0
for i in range(len(input_data_efs)):
    un_match = False
    id_ = str(input_data_efs.loc[i, 'studyid'])
    if input_data_efs.loc[i, 'r_a_typ1']:
        a_sim = input_data_efs.loc[i, 'r_a_typ1'] - 1
        b_sim = input_data_efs.loc[i, 'r_b_typ1'] - 1
        c_sim = input_data_efs.loc[i, 'r_c_typ1'] - 1
        q_sim = input_data_efs.loc[i, 'r_dqb1_typ1'] - 1
        r_sim = input_data_efs.loc[i, 'r_drb1_typ1'] - 1

        abcqr = a_sim + b_sim + c_sim + q_sim + r_sim
        if input_data_efs.loc[i, 'numhires10_1'] and (abcqr != input_data_efs.loc[i, 'numhires10_1']):
            un_match = True
            print(str(id_) + ' change10 they: ' + str(input_data_efs.loc[i, 'numhires10_1']) + ' my: ' + str(abcqr))

        abcr = a_sim + b_sim + c_sim + r_sim
        if input_data_efs.loc[i, 'numhires8_1'] and (abcr != input_data_efs.loc[i, 'numhires8_1']):
            un_match = True
            print(str(id_) + ' change8 they: ' + str(input_data_efs.loc[i, 'numhires8_1']) + ' my: ' + str(abcr))

        abc = a_sim + b_sim + c_sim
        if input_data_efs.loc[i, 'numlores6_1'] and (abc != input_data_efs.loc[i, 'numlores6_1']):
            un_match = True
            print(str(id_) + ' change6abc they: ' + str(input_data_efs.loc[i, 'numlores6_1']) + ' my: ' + str(abc))

        abr = a_sim + b_sim + r_sim
        if input_data_efs.loc[i, 'numhires6_1'] and (abr != input_data_efs.loc[i, 'numhires6_1']):
            un_match = True
            print(str(id_) + ' change6abr they: ' + str(input_data_efs.loc[i, 'numhires6_1']) + ' my: ' + str(abr))

        if input_data_efs.loc[i, 'himatchdqb1_1'] and (q_sim != input_data_efs.loc[i, 'himatchdqb1_1']):
            un_match = True
            print(str(id_) + ' change_q they: ' + str(input_data_efs.loc[i, 'himatchdqb1_1']) + ' my: ' + str(q_sim))

        # if un_match:
        #   count_unmatch+=1
        #  input_data_efs.loc[i, 'r_a_typ1'] = np.nan
print('finish check impute res')
print(count_unmatch)
# input_data_efs = input_data_efs[input_data_efs['r_a_typ1'].notna()]
print(len(input_data_efs))

###finish check impute res
input_data_efs = input_data_efs[input_data_efs['dpb1permissive'].notna()]
input_data_efs = input_data_efs.drop(['himatchdqb1_1', 'studyid', 'numlores8_1', 'numhires6_1', 'numlores6_1'],
                                     axis=1)  # 'numhires10_1'

list_unfull_numerical_column = ['karnofraw', 'numhires10_1', "coorgscore"]
input_data_efs = input_data_efs.loc[input_data_efs['numhires8_1'].isin([8])]  # stay just 8/8 match
# input_data_efs = input_data_efs.loc[input_data_efs['disease'].isin([10])  ]#stay just AML disease
input_data_efs = input_data_efs.loc[input_data_efs['txnum'].isin([1])]
# input_data_efs = input_data_efs.drop(['disease'],axis=1)

# remove features that appear in less than half the data
"""half_data_len = len(input_data_efs)/2
for column in input_data_efs:
    stat = input_data_efs[column].describe()
    h = stat['count']

    if int(stat['count']) < half_data_len:
        if not column in list_allels_count and not column in list_y :
            input_data_efs = input_data_efs.drop([column], axis=1)"""

list_unfull_continues = ['indxtx', 'median_income', 'r_a_typ2', 'r_b_typ2', 'r_c_typ2',
                         'urdbmpbdage']  # , 'impr_drb1_2','impr_dqb1_2']

list_to_one_hot = ['disease']  # 'disease','gvhprhrx' #aldist
# ('indcycle', 99, 'one-hot'),('mdstfaml', 99, 'one-hot'),('cytogeneeln', 99, 'one-hot'),('amlrxrel', 99, 'one-hot'),('aldist', 99, 'one-hot'),
list_column_with_unkwon = [('tbigp', 99, 'one-hot'), ('karnofcat', 99, 'one-hot'), ('hctcigpb', 99, 'median'),
                           ('gvhprhrxgp', 99, 'one-hot'), ('bmpbdsex', 99, 'one-hot'), ('urdbmpbdrace', 99, 'one-hot'),
                           ('racegp', 99, 'one-hot'), ('drcmvpr', 99, 'one-hot'), ('cytogene', 99, 'one-hot'),
                           ('venthxpr', 99, 'one-hot'), ('funghxpr', 99, 'one-hot'), ('condclas', 99, 'one-hot')]
###
for pair in list_column_with_unkwon:
    column = pair[0]
    char = pair[1]
    if (type(char) == list):
        for c in char:
            input_data_efs[column] = input_data_efs[column].apply(lambda x: np.nan if x == c else x)
    else:
        input_data_efs[column] = input_data_efs[column].apply(lambda x: np.nan if x == char else x)
    if pair[2] == 'median':
        list_unfull_numerical_column.append(column)
    elif pair[2] == 'one-hot':
        list_to_one_hot.append(column)
    elif pair[2] == 'mean':
        list_unfull_continues.append(column)
    else:
        print(column)

##remove post features
for feature in list_to_remove:
    if not feature in list_importent and feature in input_data_efs.columns:
        input_data_efs = input_data_efs.drop([feature], axis=1)
input_data_efs.to_excel('new_data_after_remove_ids_ALL.xlsx')

list_very_unfull_columns = []  #
# add column to indicate if have information or not
for column in list_very_unfull_columns:
    input_data_efs[column + '_is_found'] = input_data_efs[column].apply(lambda x: 1 if not pd.isnull(x) else 0)

# convert caterorial of more than 2, to one hot
for column in list_to_one_hot:
    if column in input_data_efs.columns:
        input_data_efs = pd.concat([input_data_efs, pd.get_dummies(input_data_efs[column], prefix=column)], axis=1)
        # now drop the original column
        input_data_efs.drop([column], axis=1, inplace=True)

# convert caterorial of 2 fields to 0-1
"""list_to_zero_one = []#['r_sex']
for column in list_to_zero_one:
    stat = input_data_efs[column].describe()
    char = stat['top'] #take just 1 key
    input_data_efs[column] = input_data_efs[column].apply(lambda x: 1 if x == char else 0)"""

data_efs = input_data_efs.copy()

# data_efs = data_efs[data_efs['r_a_typ1'].notna()]
data_size = len(data_efs) - 500

list_unfll = list_unfull_numerical_column + list_very_unfull_columns
# add median in unkwown valuse in continues columns
for column in list_unfll:
    if column in input_data_efs.columns:
        median = data_efs[:int(data_size * 0.8)][column].median()
        data_efs[column].fill_na(median, inplace=True)

# add mean in unkwown valuse in continues columns
for column in list_unfull_continues:
    if column in input_data_efs.columns:
        mean = data_efs[:int(data_size * 0.8)][column].mean()
        data_efs[column].fill_na(mean, inplace=True)

train_data = data_efs[:int(data_size * 0.8)]
# test_data = data_efs[int(data_size*0.7):]
valid_data = data_efs[int(data_size * 0.8):data_size]
test_data = data_efs[data_size:]
# print(len(train_data))
path = 'created_data_ALL_importatnt_model'  # 'created_data'
os.makedirs(path, exist_ok=True)
for label in list_y:  # list_y:
    df_label_train = train_data.copy(deep=True)
    df_label_valid = valid_data.copy(deep=True)
    df_label_test = test_data.copy(deep=True)
    print('\nlabel: ' + label)

    df_label_train = df_label_train[df_label_train[label].notna()]
    df_label_valid = df_label_valid[df_label_valid[label].notna()]
    df_label_test = df_label_test[df_label_test[label].notna()]
    for column in list_y:
        if label != column:
            df_label_train.drop([column], axis=1, inplace=True)
            df_label_valid.drop([column], axis=1, inplace=True)
            df_label_test.drop([column], axis=1, inplace=True)

    df_label_train.to_excel(path + '/efs_train_ol_' + label + '.xlsx')
    df_label_test.to_excel(path + '/efs_test_ol_' + label + '.xlsx')
    df_label_valid.to_excel(path + '/efs_valid_ol_' + label + '.xlsx')

train_data.to_excel(path + '/efs_train_ol_all.xlsx')
# test_data = data_efs[int(data_size*0.7):]
valid_data.to_excel(path + '/efs_valid_ol_all.xlsx')
test_data.to_excel(path + '/efs_test_ol_all.xlsx')
