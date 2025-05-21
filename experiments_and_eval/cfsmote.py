from __future__ import annotations

import random, datetime, math
from river import base, drift, neighbors
from river.forest import adaptive_random_forest
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from tree.two_faht import HoeffdingAdaptiveTreeClassifier as FAFAHT
from tree.faht import HoeffdingTreeClassifier as FAHT
from tree.feat import HoeffdingAdaptiveTreeClassifier as FEAT


class CFSMOTE(base.Classifier):

    """ CSFMOTE - fairness-aware SMOTE variant for imbalanced streams

    Parameters
    ------------

    sensitive_attribute: str that contains the name of the sensitive attribute/feature

    deprived_val: str or int that contains the deprived sensitive attribute value

    undeprived_vals = list (of str or int) which contains the non-deprived sensitive attribute values

    model_name: The classifier that gets trained, default = None -> ARF

    num_neighbours: Number of neighbours for SMOTE, default = 5

    minority_threshold: Threshold for minorty class samples, default = 0.245 (MUST not be bigger than 0.25, as we have FOUR groups; default allows for a little leeway ot be more efficient)

    min_size_allowed: Minimum number of samples in the minority class for appling SMOTE, default = 100

    disable_drift_detection: Flag which indicates whether error rate should be monitored and model reset for drift, default = False (drift IS detected)

    random_seed: ranodm seed for random, default: None

    categorical_features: Dict with keys as feature names and values as list of categories, default = None

    model_categories: List of features that are categorical (and not numerical); used for base learner, default = None

    min_situation_test: Minimum number of samples seen before situation testing, default = 500

    """

    def __init__(self, 
                sensitive_attribute: str,
                deprived_val: str | int,
                undeprived_vals = list, 
                model_name=None,
                num_neighbours=5,
                minority_threshold=0.245,
                min_size_allowed=100,
                disable_drift_detection=False,  
                random_seed=None,
                categorical_features=None,
                model_categories=None,
                min_situation_test=500):

        if categorical_features is None:
            self.categorical_features = {}
            self.categories_unknown = True
        else:
            self.categorical_features = categorical_features
            self.categories_unknown = False

        self.categories = model_categories

        self.sensitive_attribute = sensitive_attribute
        self.deprived_val = deprived_val
        self.undeprived_vals = undeprived_vals

        self.disable_drift_detection = disable_drift_detection

        self.model_name = model_name
        self.model = self.set_model()
        
        self.num_neighbours = num_neighbours #needs to be >= 1
        self.minority_threshold = minority_threshold #Should be between 0.1 and 0.5
        self.min_size_allowed = min_size_allowed #should be >= 10

        self.adwin_class = drift.ADWIN()
        self.adwin_fair = drift.ADWIN()
        if not self.disable_drift_detection:
            self.drift_detector = drift.ADWIN()

        self.W = []

        self.minority_deprived = []
        self.majority_deprived = []
        self.minority_undeprived = []
        self.majority_undeprived = []

        self.n_minority_total_deprived = 0
        self.n_majority_total_deprived = 0
        self.n_minority_total_undeprived = 0
        self.n_majority_total_undeprived = 0

        self.n_minority_generated_total_deprived = 0
        self.n_majority_generated_total_deprived = 0
        self.n_minority_generated_total_undeprived = 0
        self.n_majority_generated_total_undeprived = 0

        self.instance_generated = {}
        self.already_used = []
        self.index_values = []

        self.random_seed = random_seed

        self.count = 0
        self.min_situation_test = min_situation_test
        self.situation_tester = HoeffdingAdaptiveTreeClassifier() #simple model for situation testing, Hoeffding Trees are simple and good for streams

        #for improving distance function
        self.numerical_features = {}



    #set learner from name as string; for now only supports ARF but can easily be extended
    def set_model(self):

        drift_detector = drift.ADWIN(delta=0.002)

        if self.model_name is None:
            return adaptive_random_forest.ARFClassifier(nominal_attributes=self.categories, drift_detector=drift_detector)
        elif self.model_name == 'ARF':
            return adaptive_random_forest.ARFClassifier(nominal_attributes=self.categories, drift_detector=drift_detector)
        elif self.model_name == 'HoeffdingAdaptiveTree':
            return HoeffdingAdaptiveTreeClassifier(nominal_attributes=self.categories, drift_detector=drift_detector)
        elif self.model_name == 'HoeffdingTree':
            return HoeffdingTreeClassifier(nominal_attributes=self.categories)
        elif self.model_name == 'FEAT':
            return FEAT(nominal_attributes=self.categories, drift_detector=drift_detector, sensitive_attribute=self.sensitive_attribute, deprived_idx=self.deprived_val)
        elif self.model_name == 'FAHT':
            return FAHT(nominal_attributes=self.categories, sensitive_attribute=self.sensitive_attribute, deprived_idx=self.deprived_val)
        #TODO: Potentially add different options for models
        else:
            print("Can not use the specified classifier " + str(self.model_name) +", using ARF instead")
            return adaptive_random_forest.ARFClassifier(nominal_attributes=self.categories, drift_detector=drift_detector)


    
    def reset(self):
        self.model = self.set_model()

        self.W = []
        self.adwin_class = drift.ADWIN()
        self.adwin_fair = drift.ADWIN()

        if not self.disable_drift_detection:
            self.drift_detector = drift.ADWIN()

        self.minority_deprived = []
        self.majority_deprived = []
        self.minority_undeprived = []
        self.majority_undeprived = []

        self.n_minority_total_deprived = 0
        self.n_majority_total_deprived = 0
        self.n_minority_total_undeprived = 0
        self.n_majority_total_undeprived = 0

        self.n_minority_generated_total_deprived = 0
        self.n_majority_generated_total_deprived = 0
        self.n_minority_generated_total_undeprived = 0
        self.n_majority_generated_total_undeprived = 0

        self.instance_generated = {}
        self.already_used = []
        self.index_values = []

    #implement prediction - just use underlying classifier
    def predict_one(self, x):
        return self.model.predict_one(x)
    
    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)
    
    def learn_one(self, x, y):
        self.count += 1
        self.situation_tester.learn_one(x, y)

        situation_test_passed = True

        #if enough samples have been seen, do a situation test to see if sens. attribute does matter
        if(self.count >= self.min_situation_test):
            situation_test_passed = self.situation_test(x)

        #only use sample if situation test is passed
        if(situation_test_passed):
            self.model.learn_one(x, y)
            self.fill_batches(x, y)

        if self.categories_unknown:
            #update list of categorical features for SMOTE to use later if they are not passed to the algorithm in the beginning
            for key in x.keys():
                if not (isinstance(x[key], float) or isinstance(x[key], int) or isinstance(x[key], datetime.datetime)):
                    if not(key in self.categorical_features.keys()):
                        self.categorical_features[key] = [x[key]]
                    elif not(x[key] in self.categorical_features[key]):
                        self.categorical_features[key].append(x[key])
            

        #update Fair ADWIN with the sensitive attribute distribution
        if (x[self.sensitive_attribute] == self.deprived_val):
            self.adwin_fair.update(1.0)
        else:
            self.adwin_fair.update(0.0)

        #update Class ADWIN
        self.adwin_class.update(float(y))

        self.check_adwin_width()

        allow_SMOTE = False


        #Check if we have enough samples to allow SMOTE

        min_len = min([len(self.majority_deprived), len(self.majority_undeprived), len(self.minority_deprived), len(self.minority_undeprived)])
        
        if(min_len > self.min_size_allowed):
            allow_SMOTE = True
        

        if allow_SMOTE:
            while (self.minority_threshold > self.calculate_ratio()): #check if SMOTE is necessary
                new_instances = self.online_SMOTE() #generate new sample with SMOTE

                for new_instance in new_instances:
                    if new_instance is not None:
                        situation_test_passed = True

                        #if enough samples have been seen, do a situation test to see if sens. attribute does matter
                        if(self.count >= self.min_situation_test):
                            situation_test_passed = self.situation_test(new_instance[0])

                        if situation_test_passed:
                            self.model.learn_one(new_instance[0], new_instance[1]) #train model on SMOTE generated instance
            self.already_used = []

            
        

        #Drift detection
        if not self.disable_drift_detection:
            pred = self.model.predict_one(x)
            error_old = self.drift_detector.estimation
            if not pred is None and (float(y) == float(pred)):
                self.drift_detector.update(1.0)
            else:
                self.drift_detector.update(0.0)
            if self.drift_detector.drift_detected:
                if(self.drift_detector.estimation > error_old): #what does estimation return? error or accuracy? TODO:Check
                    print("Model Reset")
                    self.model = self.set_model() #reset model if drift was detected
                self.drift_detector = drift.ADWIN()

        
        return

    def situation_test(self, x):
        x_deprived = x.copy()
        x_undeprived = x.copy()

        x_deprived[self.sensitive_attribute] = self.deprived_val
        x_undeprived[self.sensitive_attribute] = random.choice(self.undeprived_vals)

        pred_deprived = self.situation_tester.predict_one(x_deprived)
        pred_undeprived = self.situation_tester.predict_one(x_undeprived)

        return (pred_deprived == pred_undeprived)
    
    #add instances to window W
    def fill_batches(self, x, y):
        instance = (x, float(y))
        self.W.append(instance)

        if (float(y) == 1.0):
            if (x[self.sensitive_attribute] == self.deprived_val):
                self.minority_deprived.append(instance)
                self.n_minority_total_deprived += 1
            else:
                self.minority_undeprived.append(instance)
                self.n_minority_total_undeprived += 1
            
        else:
            if (x[self.sensitive_attribute] == self.deprived_val):
                self.majority_deprived.append(instance)
                self.n_majority_total_deprived += 1
            else:
                self.majority_undeprived.append(instance)
                self.n_majority_total_undeprived += 1

    def check_adwin_width(self):
        if (self.adwin_fair.drift_detected or self.adwin_class.drift_detected):
            window_size = len(self.W)
            new_width = min(int(self.adwin_class.width), int(self.adwin_fair.width)) #take width of the smaller ADWIN window
            diff = int(window_size - new_width)

            for i in range(diff):
                #remove the old instance and adjust lists of majority and minority
                instance_removed = self.W.pop(0)
                if (instance_removed[1] == 1.0):
                    if instance_removed[0][self.sensitive_attribute] == self.deprived_val:
                        self.minority_deprived.pop(0)
                        self.n_minority_total_deprived -= 1

                        instance_key = (tuple(instance_removed[0].items()), instance_removed[1])

                        if instance_key in self.instance_generated.keys():
                            self.n_minority_generated_total_deprived -= self.instance_generated.pop(instance_key)
                    else:
                        self.minority_undeprived.pop(0)
                        self.n_minority_total_undeprived -= 1

                        instance_key = (tuple(instance_removed[0].items()), instance_removed[1])

                        if instance_key in self.instance_generated.keys():
                            self.n_minority_generated_total_undeprived -= self.instance_generated.pop(instance_key)
                else:
                    if instance_removed[0][self.sensitive_attribute] == self.deprived_val:
                        self.majority_deprived.pop(0)
                        self.n_majority_total_deprived -= 1

                        instance_key = (tuple(instance_removed[0].items()), instance_removed[1])

                        if instance_key in self.instance_generated.keys():
                            self.n_majority_generated_total_deprived -= self.instance_generated.pop(instance_key)
                    else:
                        self.majority_undeprived.pop(0)
                        self.n_majority_total_undeprived -= 1

                        instance_key = (tuple(instance_removed[0].items()), instance_removed[1])

                        if instance_key in self.instance_generated.keys():
                            self.n_majority_generated_total_undeprived -= self.instance_generated.pop(instance_key)


    def calculate_ratio(self):
        ratio = 0.0

        #calculate number of samples for each group
        n_maj_d = self.n_majority_total_deprived + self.n_majority_generated_total_deprived
        n_maj_un = self.n_majority_total_undeprived + self.n_majority_generated_total_undeprived
        n_min_d = self.n_minority_total_deprived + self.n_minority_generated_total_deprived
        n_min_un = self.n_minority_total_undeprived + self.n_minority_generated_total_undeprived

        #sum up for total
        total = (n_maj_d + n_maj_un + n_min_d + n_min_un)

        #find smallest group
        smallest_group = min(n_maj_d, n_maj_un, n_min_d, n_min_un)
        
        ratio = float(smallest_group/ total)


        return ratio
    
    #introduce a new instance - wrapper for class choice
    def online_SMOTE(self):
        new_instances = []

        #calculate number of samples for each group
        n_maj_d = self.n_majority_total_deprived + self.n_majority_generated_total_deprived
        n_maj_un = self.n_majority_total_undeprived + self.n_majority_generated_total_undeprived
        n_min_d = self.n_minority_total_deprived + self.n_minority_generated_total_deprived
        n_min_un = self.n_minority_total_undeprived + self.n_minority_generated_total_undeprived

        #sum up for total
        total = (n_maj_d + n_maj_un + n_min_d + n_min_un)

        #calculate_ratios
        r_maj_d = n_maj_d / total
        r_maj_un = n_maj_un / total
        r_min_d = n_min_d / total
        r_min_un = n_min_un / total

        #for each group that is NOT biggest, apply SMOTE
        if(r_min_d < self.minority_threshold):
            new_instance = self.generate_new_instance(self.minority_deprived)
            if new_instance is not None:
                self.n_minority_generated_total_deprived += 1
                new_instances.append(new_instance)

        if(r_min_un < self.minority_threshold):
            new_instance = self.generate_new_instance(self.minority_undeprived)
            if new_instance is not None:
                self.n_minority_generated_total_undeprived += 1
                new_instances.append(new_instance)

        if(r_maj_d < self.minority_threshold):
            new_instance = self.generate_new_instance(self.majority_deprived)
            if new_instance is not None:
                self.n_majority_generated_total_deprived += 1
                new_instances.append(new_instance)

        if(r_maj_un < self.minority_threshold):
            new_instance = self.generate_new_instance(self.majority_undeprived)
            if new_instance is not None:
                self.n_majority_generated_total_undeprived += 1
                new_instances.append(new_instance)


        return new_instances

    #actually generate a new instance from the minorty class
    def generate_new_instance(self, samples):
        random.seed(None)

        choices = list(range(0, len(samples)))
        pos = random.choice(choices)

        while pos in self.already_used and len(choices) > 1:
            choices.remove(pos)
            pos = random.choice(choices)


        self.already_used.append(pos)

        if (len(self.already_used) == len(samples)):
            self.already_used = []

        instance = samples[pos]

        k = min(len(samples), self.num_neighbours)
        dist_func = self.dist_func

        NN_alg = neighbors.LazySearch(window_size=len(samples), dist_func=dist_func) #SWINN takes absurdly long

        for n in samples:
            NN_alg.update(n)

        
        neighbours = NN_alg.search(instance, n_neighbors=k)[0]

        nn = random.randint(0, len(neighbours)-1) #maybe start with 1 because neighbours[0] is always neighbour itself

        values = {}

        for key in instance[0].keys():
            attr_instance = instance[0][key]
            attr_neighbour = neighbours[nn][0][key]

            if key=='':
                break

            #generating new sample instance for numericals and dates
            if isinstance(attr_instance, int) or isinstance(attr_instance, float) or isinstance(attr_instance, datetime.datetime):
                diff = float(attr_neighbour) - float(attr_instance)
                gap = random.random()
                values[key] = float(attr_instance) + gap*diff

            #generating new sample that is categorical (just the one that occurs most frequently in instance + neighbours)
            else:
                counts = {}
                for category in self.categorical_features[key]: #initialize count for all attribute values with zero
                    counts[category] = 0
                counts[attr_instance] += 1
                for neighbour in neighbours:
                    attr = neighbour[0][key]
                    counts[attr] += 1

                values[key] = max(counts, key=counts.get)

        new_instance = (values, instance[1])


        #update counter for how often a sample has been generated based on this instance

        instance_key = (tuple(instance[0].items()), instance[1])

        if instance_key in self.instance_generated.keys():
            #print("instance already seen")
            self.instance_generated[instance_key] += 1
        else:
            self.instance_generated[instance_key] = 1

        return new_instance

    #modified minkowski distance for including categorical things 
    def dist_func(self, aa, bb, p=2):

        a, a_y = aa
        b , b_y = bb 
        summ = 0

        for k in {*a.keys(), *b.keys()}:

            if (isinstance(a[k], str) or isinstance(b[k], str)):
                if not(a[k]==b[k]):
                    summ += 1 #distance of one for non-matching strings
            else:

                #getting norm-factor
                if k in self.numerical_features.keys():
                    if(min(a[k], b[k]) < self.numerical_features[k]['min']):
                        self.numerical_features[k]['min'] = min(a[k], b[k])
                    if(max(a[k], b[k]) > self.numerical_features[k]['max']):
                        self.numerical_features[k]['max'] = max(a[k], b[k])

                else:
                    new_dict={}
                    new_dict['min'] = min(a[k], b[k])
                    new_dict['max'] = max(a[k], b[k])
                    self.numerical_features[k] = new_dict


                diff = abs(a[k] - b[k])
                max_diff = abs(self.numerical_features[k]['min'] - self.numerical_features[k]['max']) + 0.000000001 #avoid zero div
                summ += ((diff/max_diff))**p #norming
                
        
        return summ**(1/p)
