class ClassificationFlow:
    """Handle classification of new cases.

    Using instance variables from SingleAlgorithmCase classifies all new user-input cases.
    """

    def __init__(self, single_algorithm_case):
        self.studied_case = single_algorithm_case
        self.user_provided_test_case = None
        self.user_provided_test_attribute = None
        self.current_attribute_label = None
        self.test_case_class = None
        self.decision_tree = self.studied_case.decision_tree
        self.attribute_possible_values = self.decision_tree.master_set_values
        self.values_order = dict()

    def run_user_classification_loop(self):
        self.prepare_values_order()
        print("Press q to quit during classification")
        while True:
            try:
                self.user_provided_test_case = self.provide_test_case()
                print("Classifying...")
                self.classify()
                print("Test case: " + str(self.user_provided_test_case) + " belongs to class: " + self.test_case_class)
            except UserQuitException:
                print("Goodbye")
                return

    def prepare_values_order(self):
        for index, attribute in enumerate(self.studied_case.training_labels[:-1]):
            self.values_order[attribute] = index

    def provide_test_case(self):
        single_test_case = []
        for attribute_label in self.studied_case.training_labels[:-1]:
            self.current_attribute_label = attribute_label
            self.provide_test_attribute()
            single_test_case.append(self.user_provided_test_attribute)
        print(single_test_case)
        return single_test_case

    def provide_test_attribute(self):
        chosen_test_attribute = input("Input value for attribute named: " + self.current_attribute_label + ". "
                                      + "Possible values are: "
                                      + str(self.attribute_possible_values[self.current_attribute_label]))
        if chosen_test_attribute in self.attribute_possible_values[self.current_attribute_label]:
            print("Chosen value " + chosen_test_attribute + " for attribute " + self.current_attribute_label)
            self.user_provided_test_attribute = chosen_test_attribute
            return
        else:
            if chosen_test_attribute == 'q':
                raise UserQuitException
            UserInputEvaluation.retry_user_input(self.provide_test_attribute, "No such attribute value in training data")

    def classify(self):
        current_node = self.decision_tree.tree_root
        while len(current_node.children) != 0:
            current_attribute = current_node.node_name
            for child in current_node.children:
                if child.label_on_branch_before == self.user_provided_test_case[self.values_order[current_attribute]]:
                    current_node = child
                    break
        self.test_case_class = current_node.node_name


class UserQuitException(Exception):
    """Exception raised when user inputs 'q' character. Used to break out of classification flow"""
    pass


class UserInputEvaluation:
    @staticmethod
    def retry_user_input(prompt_function, why_retry_message):
        print(why_retry_message)
        prompt_function()