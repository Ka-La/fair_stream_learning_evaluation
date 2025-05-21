from data_synthesizer.DataDescriber import DataDescriber
from data_synthesizer.DataGenerator import DataGenerator
from data_synthesizer.ModelInspector import ModelInspector
from data_synthesizer.lib.utils import read_json_file, display_bayesian_network
from data_synthesizer.lib.PrivBayes import construct_noisy_conditional_distributions
from pathlib import Path

import pandas as pd

description = f'/vol/kiakademie/stream_fairness_exp/student_performance_from_survey_description_full_por.json'
clean_description = f'/vol/kiakademie/stream_fairness_exp/student_performance_from_survey_description_full_por_debiased.json'

clean_data = f'/vol/kiakademie/stream_fairness_exp/student_performance_from_survey_debiased.csv'
unmodified_data = f'/vol/kiakademie/stream_fairness_exp/student_performance_from_survey_unmodified.csv'

num_tuples_to_generate = 5000

from DataManipulator import DataManipulator

sensitive_atrributes = ['sex']

manipulator = DataManipulator()

manipulator.clean_dataset_description_save_structure(description, clean_description, sensitive_atrributes)

display_bayesian_network(manipulator.bayesian_network)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, clean_description)
generator.save_synthetic_data(clean_data)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description)
generator.save_synthetic_data(unmodified_data)