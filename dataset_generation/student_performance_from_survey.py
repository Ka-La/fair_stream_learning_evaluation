from data_synthesizer.DataDescriber import DataDescriber
from data_synthesizer.DataGenerator import DataGenerator
from data_synthesizer.ModelInspector import ModelInspector
from data_synthesizer.lib.utils import read_json_file, display_bayesian_network
from data_synthesizer.lib.PrivBayes import construct_noisy_conditional_distributions
from pathlib import Path

import pandas as pd

# input dataset
input_data = './student_performance.csv'

# location of the output files

description = f'./student_performance_from_survey_description_full_por.json'

bayes_net = [["sex", ["activities", "famsup"]],
["schoolsup", ["sex"]],
["school", ["schoolsup"]],
["G1", ["school", "sex"]],
["address", ["school"]],
["reason", ["school"]],
["failures", ["G1"]],
["G2", ["G1"]],
["G3", ["G1", "G2"]],
["age", ["failures"]],
["higher", ["G1", "age"]],
["guardian", ["age"]],
["romantic", ["age"]],
["Pstatus", ["guardian"]],
["famsize", ["Pstatus"]],
["nursery", ["famsize"]],
["traveltime", ["address"]],
["Mjob", ["internet"]],
["Medu", ["Mjob"]],
["Fedu", ["Medu"]],
["Fjob", ["Fedu"]],
["internet", ["school", "address"]]]

categorical_attributes = {'school': True, 'sex': True, 'age': True, 'address': True, 'famsize': True, 'Pstatus': True, 'Medu': True, 'Fedu': True, 'Mjob': True, 'Fjob': True, 'guardian': True, 'reason': True,
                          'traveltime': True, 'failures': True, 'schoolsup': True, 'paid': True, 'activities': True, 'nursery': True, 'higher': True, 'internet': True, 'famsup': True, 'romantic': True, 'goout': True,}
epsilon = 0.0

describer = DataDescriber()
describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        attribute_to_is_categorical=categorical_attributes)
describer.df_encoded = describer.encode_dataset_into_binning_indices()
if describer.df_encoded.shape[1] < 2:
            raise Exception("Correlated Attribute Mode requires at least 2 attributes(i.e., columns) in dataset.")
describer.bayesian_network = bayes_net
describer.data_description['bayesian_network'] = describer.bayesian_network
print("Bayes net done")
describer.data_description['conditional_probabilities'] = construct_noisy_conditional_distributions(
    describer.bayesian_network, describer.df_encoded, epsilon / 2)
    
print("Distribution done")
display_bayesian_network(describer.bayesian_network)
describer.save_dataset_description_to_file(description)