import pandas as pd
import datetime
from river import stream, evaluate, tree
import river.metrics
from river_fairness_metrics.metrics import Metrics as FMetrics
from river_fairness_metrics.equalized_odds import Equalized_FPR
from river_fairness_metrics.equal_opportunity import Equal_Opportunity
from river_fairness_metrics.demographic_parity import Demographic_Parity
from river_fairness_metrics.disparate_impact import Disparate_Impact
from tree.two_faht import HoeffdingAdaptiveTreeClassifier





def run_exp_fairness_adult(file_in, file_out):

    converter = {'age': float, 'income': float, 'education-num': float, 'capital-gain': float, 'capital-loss': float, 'hours-per-week': float}


    category_list=['education','workclass','marital_status','occupation', 'relationship', 'race', 'native_country']


    drift_window_threshold = 23000
    drift_t = 44501

    model = HoeffdingAdaptiveTreeClassifier(nominal_attributes=category_list, split_criterion='flex_info_gain', sensitive_attribute="sex", deprived_idx="Female", drift_window_threshold=drift_window_threshold, drift_t=drift_t)

    
    X_y = stream.iter_csv(file_in, target="income", converters=converter)

    sens_att = ("sex", "Female")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))


    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []

    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []


    results['time'] = []
    

    for x, y in X_y:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        if y_pred is not None:

            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())


            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)


    
def run_test_fairness(mode, repetition):


    print("Run " + str(repetition))

    in_file = f"./data/Clean_Adult_Dataset/run_{repetition}.csv"
    out_file = f"./data/control_clean/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    '''


    in_file = f"./data/synth_adult/drift/clean_female_raise_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_clean_female_raise/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    #####

    in_file = f"./data/synth_adult/drift/original_female_raise_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_original_female_raise/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####

    in_file = f"./data/synth_adult/drift/original_raise_25000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_raise/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####

    in_file = f"./data/synth_adult/drift/original_aging_15000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_aging/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####


    in_file = f"./data/synth_adult/drift/original_trans_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_original_trans/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####

    in_file = f"./data/synth_adult/drift/original_to_clean_trans_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_original_to_clean_trans/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####

    in_file = f"./data/synth_adult/drift/clean_trans_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_clean_trans/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)

    ####
    
    in_file = f"./data/synth_adult/drift/clean_to_original_trans_20000/run_{repetition}.csv"
    out_file = f"./data/synth_adult/drift/results_clean_to_original_trans/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)
    '''

    
def run_test_repeatedly(mode, repetitions):

    for i in range(repetitions):
        run_test_fairness(mode, i)



if __name__=='__main__':
    run_test_repeatedly('2faht', 10)