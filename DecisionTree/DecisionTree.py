from graphviz import Digraph
import pydotplus
import math
import csv
import random
import numpy as np

file_name = 'onlinefraud.csv'


######################## global variables ##############################
min_amount = 0.0; max_amount = 0.0  
min_old_bal_org = 0.0; max_old_bal_org = 0.0
min_new_bal_org = 0.0; max_new_bal_org = 0.0
min_old_bal_dst = 0.0; max_old_bal_dst = 0.0
min_new_bal_dst = 0.0; max_new_bal_dst = 0.0
#########################################################################




#########################################################################
#   this function used for having all
#   possible values of every attribute :
all_attr_vals = {}
def make_all_attr_vals(data):
    for row, label in data:
        for attr, val in row.items():
            if attr not in all_attr_vals:
                all_attr_vals[attr] = set()
            all_attr_vals[attr].add(val)
#########################################################################




#########################################################################
#   these functions are for calculating
#   the information gain from entropy :
def entropy(data) :
    num_data = len(data)
    if num_data == 0:
        return 0

    num_positives = sum(label for _, label in data)
    num_negatives = num_data - num_positives

    if num_positives == 0 or num_negatives == 0:
        return 0

    p_positive = num_positives / num_data
    p_negative = num_negatives / num_data
    
    result = -p_positive * math.log2(p_positive) - p_negative * math.log2(p_negative)
    return result

def IG_entropy(data, attribute):
    entropy_before = entropy(data)
    num_data = len(data)
    entropy_after = 0.0

    values = set(all_attr_vals[attribute])
    for value in values:
        subset = [(row, label) for row, label in data if row[attribute] == value]
        weight = len(subset) / num_data
        entropy_after += weight * entropy(subset)

    information_gain = entropy_before - entropy_after
    return information_gain
#########################################################################



#########################################################################
#   these functions are for calculating
#   the information gain from gini idex :
def gini_index(data):
    num_data = len(data)
    if num_data == 0:
        return 0

    num_positives = sum(label for _, label in data)
    num_negatives = num_data - num_positives

    p_positive = num_positives / num_data
    p_negative = num_negatives / num_data

    gini = 1 - (p_positive ** 2 + p_negative ** 2)
    return gini

def gini_impurity(data, attribute):
    num_data = len(data)
    weighted_gini_after = 0.0

    values = set(all_attr_vals[attribute])
    for value in values:
        subset = [(x, y) for x, y in data if x[attribute] == value]
        weight = len(subset) / num_data
        weighted_gini_after += weight * gini_index(subset)

    return weighted_gini_after

def IG_gini(data, attribute):
    gini_before = gini_index(data)
    gini_impurity_before = gini_impurity(data, attribute)
    information_gain = gini_before - gini_impurity_before
    return information_gain
#########################################################################




#########################################################################
#   these functions are for testing data
#   with both DecisionTree made by gini and entropy :
def predict(tree, row):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        # value = row.get(attribute)
        value = row[attribute]

        if value in tree[attribute]:
            subtree = tree[attribute][value]
            return predict(subtree, row)
        else:
            return None
    else:
        return tree

def test_data(gini_tree, entropy_tree, test):
    num_test = len(test)
    gini_count = 0
    entropy_count = 0

    for row, label in test:
        gini_test = predict(gini_tree, row)
        if not gini_test is None:
            if gini_test == label:
                gini_count += 1

        entropy_test = predict(entropy_tree, row)
        if not entropy_test is None:
            if entropy_test == label:
                entropy_count += 1

    gini_acc = (float(gini_count) / num_test) * 100
    entropy_acc = (float(entropy_count) / num_test) * 100
    return gini_acc, entropy_acc
#########################################################################




#########################################################################
#   these functions are for creating the
#   DecisionTree from data :
def select_best_attribute(data, attributes, criterion):
    if criterion == 'gini':
        crit_func = IG_gini
    elif criterion == 'entropy':
        crit_func = IG_entropy
    else:
        raise ValueError("Invalid criterion")

    best_info_gain = -1
    best_attribute = None

    for attribute in attributes:
        info_gain = crit_func(data, attribute)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

def Build_tree(data, attributes, criterion):
    labels = [label for _, label in data]

    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not attributes:
        return max(set(labels), key=labels.count)

    best_attr = select_best_attribute(data, attributes, criterion)

    tree = {best_attr: {}}
    attributes.remove(best_attr)

    values = set(all_attr_vals[best_attr])
    for value in values:
        subset = [(row, label) for row, label in data if row[best_attr] == value]

        if not subset:
            tree[best_attr][value] = max(set(labels), key=labels.count)
        else:
            tree[best_attr][value] = Build_tree(subset, attributes.copy(), criterion)

    return tree
#########################################################################




#########################################################################
#   these functions are used to discreting
#   and updating datas of a single row :
def Discreting(length, value , min, max):
    return math.floor((float(value) - min) / ((max - min) / length))

def edit_row(row):
    label = row.pop('isFraud')
    row.pop('nameOrig') # this is unique 
    row.pop('nameDest')

    name = 'type'
    if   row[name] == 'PAYMENT':  row[name] = 0
    elif row[name] == 'TRANSFER': row[name] = 1
    elif row[name] == 'CASH_OUT': row[name] = 2
    elif row[name] == 'DEBIT':    row[name] = 3
    else:                         row[name] = 4 # 'CASH_IN'

    name = 'amount'
    row[name] = Discreting(6, row[name], min_amount, max_amount)

    name = 'oldbalanceOrg'
    row[name] = Discreting(8, row[name], min_old_bal_org, max_old_bal_org)

    name = 'newbalanceOrig'
    row[name] = Discreting(8, row[name], min_new_bal_org, max_new_bal_org)

    name = 'oldbalanceDest'
    row[name] = Discreting(8, row[name], min_old_bal_dst, max_old_bal_dst)

    name = 'newbalanceDest'
    row[name] = Discreting(8, row[name], min_new_bal_dst, max_new_bal_dst)

    return row, int(label)
#########################################################################




#########################################################################
#   these functions are for creating and analysing
#   data and test before making DecisionTree :
def read_csv():
    with open(file_name, 'r') as csvfile:
        rows = csv.DictReader(csvfile)
        pos_list = []
        neg_list = []


        for row in rows:            
            data = {key: value for key, value in row.items()}
            if data['isFraud'] == '1' :
                pos_list.append(data)
            else:
                neg_list.append(data)


        random.shuffle(pos_list)
        random.shuffle(neg_list)


        # selected_data = pos_list[:2] + neg_list[:98]
        # selected_data = pos_list[:10] + neg_list[:490]
        # selected_data = pos_list[:40] + neg_list[:1960]
        # selected_data = pos_list[:100] + neg_list[:4900]
        selected_data = pos_list[:211] + neg_list[:9788]
        # selected_data = pos_list[:1000] + neg_list[:49000]

        # selected_test = pos_list[-1000:-1] + neg_list[-48000:-1]
        selected_test = pos_list[-211:-1] + neg_list[-9788:-1]
        # selected_test = pos_list[-1000:-1] + neg_list[-1000:-1]

        random.shuffle(selected_data)
        random.shuffle(selected_test)

        return selected_data, selected_test

def data_analysis():
    data = np.genfromtxt(file_name, delimiter=',', skip_header=True)


    col = data[:, 2]
    global min_amount, max_amount, min_old_bal_dst, min_old_bal_org
    global max_old_bal_org, min_new_bal_org, max_new_bal_org
    global max_old_bal_dst, min_new_bal_dst, max_new_bal_dst


    min_amount = np.min(col)
    max_amount = np.max(col)


    col = data[:, 4]
    min_old_bal_org = np.min(col)
    max_old_bal_org = np.max(col)


    col = data[:, 5]
    min_new_bal_org = np.min(col)
    max_new_bal_org  = np.max(col)


    col = data[:, 7]
    min_old_bal_dst = np.min(col)
    max_old_bal_dst  = np.max(col)


    col = data[:, 8]
    min_new_bal_dst = np.min(col)
    max_new_bal_dst  = np.max(col)    

    selected_data, selected_test = read_csv()

    data = []
    for row in selected_data:
        data.append(edit_row(row))

    test = []
    for row in selected_test:
        test.append(edit_row(row))

    return data, test
#########################################################################




#########################################################################
#   this function is used to generate a tree
#   that can be visualized using graphviz :
def visualize_tree(tree):
    dot = Digraph()

    def add_nodes_edges(tree, parent=None):
        if isinstance(tree, dict):
            for key, val in tree.items():
                if parent:
                    dot.edge(parent, str(key), style='solid', color='black')
                add_nodes_edges(val, str(key))
        else:
            dot.node(str(tree), style='filled', fillcolor='cyan')

    add_nodes_edges(tree)
    return dot
#########################################################################




###############################  main  ##################################
def main():
    data, test = data_analysis()

    make_all_attr_vals(data)
    attributes = list(all_attr_vals.keys())


    gini_tree = Build_tree(data, attributes.copy(), 'gini')
    entropy_tree = Build_tree(data, attributes.copy(), 'entropy')


    gini_acc, entropy_acc = test_data(gini_tree, entropy_tree, test)

    print('gini accuracy : ', gini_acc) 
    print('entropy accuracy : ', entropy_acc)



    ################# Visualize the decision tree #################
    dot_gini_tree = visualize_tree(gini_tree)
    dot_entropy_tree = visualize_tree(entropy_tree)

    dot_gini_tree.render("gini_tree", format="pdf", cleanup=True)
    dot_entropy_tree.render("entropy_tree", format="pdf", cleanup=True)
#########################################################################




if __name__ == '__main__':
    main()