import csv
import numpy as np


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

    def create_tree(self):
        print(self.training_data, self.training_labels)
