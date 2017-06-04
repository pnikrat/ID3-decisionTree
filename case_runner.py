import csv
import math

class SingleAlgorithmCase:
    """Handle single program run.

    Imports training data with CsvFileHandler object and passes it to TreeCreator object
    to create minimal decision tree.
    """
    def __init__(self, cli_argument):
        self.file_handler = CsvFileHandler(cli_argument)
        self.training_labels = []
        self.training_data = []

    def start(self):
        self.file_handler.import_training_data()
        self.training_labels, self.training_data = self.file_handler.get_training_data_with_labels()
        if self.file_handler.successful_import:
            # use training data to create a minimal tree
            tree_creation = TreeCreator(self.training_labels, self.training_data)
            tree_creation.create_tree()
        else:
            print("File does not exist")


class CsvFileHandler:
    """Import training data from CSV file.

    Imports training data to list of lists (each list is a single training case).
    """
    training_labels = []
    training_data = []
    successful_import = True

    def __init__(self, filename):
        self.filename = filename

    def import_training_data(self):
        try:
            with open(self.filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for index, row in enumerate(csvreader):
                    if index == 0:
                        self.training_labels = row
                    else:
                        single_training_case = row
                        if len(single_training_case) is not 0:
                            self.training_data.append(single_training_case)
        except IOError:
            self.successful_import = False

    def get_training_data_with_labels(self):
        return self.training_labels, self.training_data


class TreeCreator:
    def __init__(self, training_labels, training_data):
        self.training_labels = training_labels
        self.training_data = training_data
        self.main_set_entropy = None
        self.tree_root = None

    def create_tree(self):
        self.main_set_entropy = self.calculate_entropy(self.training_data)
        if self.main_set_entropy == 0:
            print("same decisions in all set")
        else:
            partition_attribute = self.choose_attribute_for_partition(self.training_data)
            print(partition_attribute)
            # start tree creation here

    def calculate_entropy(self, cases):
        entropy = 0.0
        number_of_all_cases = len(cases)
        case_decision_dict = self.create_dict_with_number_of_occurences(cases, -1)
        for v in case_decision_dict.values():
            entropy += (-(v/number_of_all_cases) * math.log((v/number_of_all_cases), 2))
        return entropy

    def create_dict_with_number_of_occurences(self, list_to_check, index):
        dict_with_number_of_occurences = dict()
        for x in list_to_check:
            if x[index] not in dict_with_number_of_occurences.keys():
                dict_with_number_of_occurences[x[index]] = 1
            else:
                dict_with_number_of_occurences[x[index]] += 1
        return dict_with_number_of_occurences

    def choose_attribute_for_partition(self, training_data):
        total_attribute_entropy = dict()
        for index, attribute in enumerate(self.training_labels[:-1]):
            print(attribute, index)
            attribute_specific_entropy = dict()
            attribute_specific_division = self.create_dict_with_number_of_occurences(training_data, index)
            print(attribute_specific_division)
            for k,v in attribute_specific_division.items():
                key_specific_cases = [x for x in training_data if x[index] == k]
                attribute_specific_entropy[k] = self.calculate_entropy(key_specific_cases)
            print(attribute_specific_entropy)
            total_attribute_entropy[attribute] = self.calculate_weighed_entropy(attribute_specific_entropy,
                                                                                attribute_specific_division)
        total_attribute_info_gain = self.change_entropy_to_info_gain(total_attribute_entropy)
        print(total_attribute_info_gain)
        # return key with maximum value in dict
        return max(total_attribute_info_gain, key=total_attribute_info_gain.get)

    def calculate_weighed_entropy(self, entropies, weighs):
        total_entropy = 0.0
        number_of_cases = sum(weighs.values())
        for k,v in weighs.items():
            total_entropy += entropies[k] * (v/number_of_cases)
        return total_entropy

    def change_entropy_to_info_gain(self, entropies):
        entropies_copy = {k: self.main_set_entropy - v for k,v in entropies.items()}
        return entropies_copy