import json

from pathlib import Path
import numpy as np
from numpy import random
from pandas import DataFrame, read_csv
from operator import add, mul
from typing import List
import ast

from DataSynthesizer.datatypes.utils.AttributeLoader import parse_json
from DataSynthesizer.lib.utils import read_json_file, generate_random_string


class DataManipulator(object):
    def __init__(self):

        self.description = {}
        self.bayesian_network: List = None


    #create new dataset_description_file for generation where outgoing edges from sensitive attributes are completely severed
    def clean_dataset_description_radical(self, description_file, out_file, sensitive_attributes):

        self.description = read_json_file(description_file)

        bn = self.description['bayesian_network']

        mod_bn = []

        for child, parents in bn:

            new_parents = []

            for i in range(len(parents)):

                parent = parents[i]

                if(parent in sensitive_attributes):
                    print(f'Removing outgoing edge from {parent} to {child}')

                    parent_len = len(self.description['attribute_description'][parent]['distribution_bins'])
                    child_conditional_distributions = self.description['conditional_probabilities'][child]


                    #if the child has only this one parent, the distribution will be summed over only that parent
                    if (len(parents) <= 1):

                        new_dist = np.zeros(parent_len)
                        for j in range(parent_len):
                            new_dist = list(map(add, child_conditional_distributions[str([j])], new_dist))

                        new_dist = self.normalize(new_dist)
                        self.description['conditional_probabilities'][child] = new_dist



                    #else, the parent gets "summed out" over the distribution with the other prents
                    else:
                        new_cond_dist = {}

                        for parents_instance in child_conditional_distributions.keys():
                            dist = child_conditional_distributions[parents_instance]

                            parents_instance = list(eval(parents_instance))
                            del parents_instance[i] #delete the number indicating the old parent
                            parents_instance = str(parents_instance)
                        
                            if(parents_instance in new_cond_dist):
                                new_cond_dist[parents_instance] = list(map(add, dist, new_cond_dist[parents_instance]))
                            
                            else:
                                new_cond_dist[parents_instance] = dist

                        new_cond_dist = self.normalize_dict(new_cond_dist)
                        
                        self.description['conditional_probabilities'][child] = new_cond_dist

                else:
                    new_parents.append(parent)

            if(len(new_parents) > 0):
                mod_bn.append(list([child, new_parents]))

        self.description['bayesian_network'] = mod_bn
        self.bayesian_network = mod_bn
        self.save_dataset_description_to_file(out_file)


    def clean_dataset_description_save_structure(self, description_file, out_file, sensitive_attributes):

        self.description = read_json_file(description_file)

        bn = self.description['bayesian_network']
        self.bayesian_network = bn

        for child, parents in bn:

            for i in range(len(parents)):

                parent = parents[i]

                if(parent in sensitive_attributes):
                    print(f'Uncorrelating outgoing edge from {parent} to {child}')

                    parent_len = len(self.description['attribute_description'][parent]['distribution_bins'])
                    child_len = len(self.description['attribute_description'][child]['distribution_bins'])
                    
                    child_conditional_distributions = self.description['conditional_probabilities'][child]

                
                    #if the child has only this one parent, the distribution will be summed over only that parent
                    #then is normalized and that distribution again given for all values of the parent
                    if (len(parents) <= 1):

                        new_dist = np.zeros(child_len)

                        for j in range(parent_len):
                            new_dist = list(map(add, child_conditional_distributions[str([j])], new_dist))

                        new_dist = self.normalize(new_dist)

                        new_cond_dist = {}

                        for j in range(parent_len):
                            new_cond_dist[str([j])] = new_dist
                        

                        self.description['conditional_probabilities'][child] = new_cond_dist



                    #else, the parent gets "summed out" over the distribution with the other prents
                    else:
                        new_cond_dist = {}
                        final_cond_dist = {}
                        new_old_keys = {}

                        for parents_instance in child_conditional_distributions.keys():
                            dist = child_conditional_distributions[parents_instance]
                            new_parents_instance = list(eval(parents_instance))

                            del new_parents_instance[i] #delete the number indicating the old parent

                            new_old_keys[parents_instance] = str(new_parents_instance) #save connections for ease later
                            parents_instance = str(new_parents_instance)
                            

                            if(parents_instance in new_cond_dist):
                                new_cond_dist[parents_instance] = list(map(add, dist, new_cond_dist[parents_instance]))
                            
                            else:
                                new_cond_dist[parents_instance] = dist

                        new_cond_dist = self.normalize_dict(new_cond_dist)

                        for key in new_old_keys.keys():
                            new_key = new_old_keys[key]
                            final_cond_dist[key] = new_cond_dist[new_key]
                        
                        self.description['conditional_probabilities'][child] = final_cond_dist

        self.save_dataset_description_to_file(out_file)

    def get_attr_bins(self, description_file, attribute):
        self.description = read_json_file(description_file)

        attributes = self.description['attribute_description'][attribute]['distribution_bins']
        print(attributes)

    def induce_data_drift(self, description_file, out_file, changing_attributes, add_dist, non_disc=False):

        self.description = read_json_file(description_file)

        bn = self.description['bayesian_network']
        self.bayesian_network = bn



        for child in changing_attributes:

            print(f'Inducing drift for {child}')

            child_conditional_distributions = self.description['conditional_probabilities'][child]

            new_cond_dist = {}
            
            if not non_disc:
                parent_indices = add_dist[child]['parent_indices']
                parent_values = add_dist[child]['parent_values']

            for parents_instance in child_conditional_distributions.keys():
                dist = child_conditional_distributions[parents_instance]
                parents_instance = ast.literal_eval(parents_instance)
                

                
                if non_disc:
                    new_cond_dist[str(parents_instance)] = list(map(mul, dist, add_dist[child]['dist']))
                    #new_cond_dist[parents_instance] = list(map(add, dist, add_dist[child]['dist']))
                else:
                    drift_criteria_fulfilled = True
                    for x in parent_indices:
                        print(x, parent_values[x])
                        if parents_instance[x] not in parent_values[x]:
                            print(str(parents_instance[x]) + " is not used for drift because parent values are " + str(parent_values[x]))
                            drift_criteria_fulfilled = False
                    if drift_criteria_fulfilled:
                        new_cond_dist[str(parents_instance)] = list(map(mul, dist, add_dist[child]['dist']))
                        #new_cond_dist[parents_instance] = list(map(add, dist, add_dist[child]['dist']))
                    else:
                        new_cond_dist[str(parents_instance)] = dist

            new_cond_dist = self.normalize_dict(new_cond_dist)
            
            self.description['conditional_probabilities'][child] = new_cond_dist

        self.save_dataset_description_to_file(out_file)


    def save_dataset_description_to_file(self, file_name):
        Path(file_name).touch()
        with open(file_name, 'w') as outfile:
            json.dump(self.description, outfile, indent=4)

    def normalize(self, norm_list):

        normalizer = sum(norm_list)
        if normalizer == 0.0:
            print("NORMALIZING WITH ZERO!")
            print(norm_list)
            normalizer = 1.0

        normed_list = []

        for elem in norm_list:
            normed_list.append(float(elem/normalizer))

        return normed_list
    
    def normalize_dict(self, norm_dict):

        for key in norm_dict.keys():
            norm_dict[key] = self.normalize(norm_dict[key])

        return norm_dict
    
    #the attribute to reject as category and value in a tuple in rej_attr
    def induce_sampling_bias(self, from_file, to_file, rej_attr, rej_prob=0.3):

        try:
            mod_dataset = read_csv(from_file, skipinitialspace=True)
        except (UnicodeDecodeError, NameError):
            mod_dataset = read_csv(from_file, skipinitialspace=True,
                                     encoding='latin1')

        rej_name, rej_value = rej_attr

        print(f'Inducing sampling bias for {rej_name} value {rej_value} with probabiity {rej_prob}')

        drop_candidates = mod_dataset[mod_dataset[rej_name] == rej_value].index.tolist()

        # choose according to rej_prob randomly
        drop_indices = np.random.choice(drop_candidates, size = int(mod_dataset.shape[0]*rej_prob))
        # drop them
        mod_dataset.drop(index=drop_indices, inplace=True)

        Path(to_file).touch()
        mod_dataset.to_csv(to_file, index=False)

    def induce_omitted_var_bias(self, from_file, to_file, rej_name):
        try:
            mod_dataset = read_csv(from_file, skipinitialspace=True)
        except (UnicodeDecodeError, NameError):
            mod_dataset = read_csv(from_file, skipinitialspace=True,
                                     encoding='latin1')
            
        mod_dataset = mod_dataset.drop(rej_name, axis=1)
            
        Path(to_file).touch()
        mod_dataset.to_csv(to_file, index=False)


    @staticmethod
    def get_sampling_order(bn):
        order = [bn[0][1][0]]
        for child, _ in bn:
            order.append(child)
        return order
