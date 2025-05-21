from data_synthesizer.DataDescriber import DataDescriber
from data_synthesizer.DataGenerator import DataGenerator
from data_synthesizer.ModelInspector import ModelInspector
from data_synthesizer.lib.utils import read_json_file, display_bayesian_network
from data_synthesizer.lib.PrivBayes import construct_noisy_conditional_distributions
from pathlib import Path

import pandas as pd

# input dataset
input_data = './adult/Adult_Preprocessed'

# location of the output files

description = f'/vol/kiakademie/stream_fairness_exp/adult_from_survey_description_preprocessed.json'

bayes_net = [["relationship", ["sex"]],
["age", ["relationship"]],
["hours-per-week", ["sex", "age", "relationship"]],
["workclass", ["hours-per-week", "age", "relationship"]],
["occupation", ["hours-per-week", "age", "sex", "workclass"]],
["education", ["occupation", "age", "sex", "workclass"]],
["native-country", ["education", "age", "workclass"]],
["marital-status", ["native-country", "age", "relationship"]],
["race", ["native-country", "age", "sex", "relationship"]],
["capital-gain", ["relationship", "occupation", "education"]],
["capital-loss", ["relationship", "occupation","capital-gain"]],
["income",  ["relationship", "occupation","capital-gain", "capital-loss", "age", "education"]]]

categorical_attributes = {'education': True, 'workclass': True, 'marital-status' : True, 'occupation' : True, 'relationship' : True, 'race' : True, 'native-country' : True}
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