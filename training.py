import math


class TreeCreator:
    """Creates single level of decision tree

    @:param training_labels - list of attribute labels to work on
    @:param training_data - list of training cases we take into consideration on that level of tree
    @:param level - level of tree (1 is root, 2 is next level below root, etc...)
    @:param master_set_values - all available values that attributes can have. Generated only once at root
    """
    def __init__(self, training_labels, training_data, level, master_set_values):
        self.training_labels = training_labels
        self.training_data = training_data
        self.main_set_entropy = None
        self.tree_root = None
        self.level = level
        self.utils = CalculationUtils()
        self.key_specific_cases_dict = dict()
        self.master_set_values = master_set_values

    def create_tree(self):
        self.main_set_entropy = self.utils.calculate_entropy(self.training_data)
        if self.main_set_entropy == 0:  # training data with single decision special case
            self.tree_root = TreeNode(None, self.level)
            self.tree_root.node_name = self.training_data[0][-1]
        else:
            # master_set_values are used to determine when we have to add ??? nodes (??? nodes have no training data)
            if len(self.master_set_values) == 0:
                self.generate_master_set_values()
            # regular decision flow - on which attribute do we split this level?
            self.prepare_key_specific_cases_dict()
            partition_attribute = self.choose_attribute_for_partition(self.training_data)
            unclassified_nodes = self.create_tree_level(partition_attribute)
            if len(unclassified_nodes) != 0:
                # unclassified nodes are those that link to next level
                for x in unclassified_nodes:
                    training_labels_next_level = self.training_labels[:]
                    index_of_unnecessary_attribute = training_labels_next_level.index(partition_attribute)
                    training_labels_next_level.remove(partition_attribute)
                    training_data_next_level = self.trim_node_case_set(self.tree_root.children[x].node_case_set[:],
                                                                       index_of_unnecessary_attribute)
                    next_level_creator = TreeCreator(training_labels_next_level, training_data_next_level,
                                                     self.level+1, self.master_set_values)
                    # recursive call here
                    next_level_creator.create_tree()
                    # link levels to each other
                    next_level_creator.tree_root.label_on_branch_before = self.tree_root.children[x].\
                        label_on_branch_before
                    self.tree_root.children[x] = next_level_creator.tree_root

    def print_tree(self):
        print(self.tree_root)

    def prepare_key_specific_cases_dict(self):
        for x in self.training_labels[:-1]:
            self.key_specific_cases_dict[x] = dict()

    def generate_master_set_values(self):
        for index, attribute in enumerate(self.training_labels[:-1]):
            self.master_set_values[attribute] = set()
            for x in self.training_data:
                if x[index] not in self.master_set_values[attribute]:
                    self.master_set_values[attribute].add(x[index])

    def choose_attribute_for_partition(self, training_data):
        total_attribute_entropy = dict()
        for index, attribute in enumerate(self.training_labels[:-1]):
            attribute_specific_entropy = dict()
            attribute_specific_division = self.utils.create_dict_with_number_of_occurences(training_data, index)
            for k, v in attribute_specific_division.items():
                key_specific_cases = [x for x in training_data if x[index] == k]
                self.key_specific_cases_dict[attribute][k] = key_specific_cases
                attribute_specific_entropy[k] = self.utils.calculate_entropy(key_specific_cases)
            total_attribute_entropy[attribute] = self.calculate_weighed_entropy(attribute_specific_entropy,
                                                                                attribute_specific_division)
        total_attribute_info_gain = self.change_entropy_to_info_gain(total_attribute_entropy)
        # return key with maximum value in dict - this key is attribute we partition on
        return max(total_attribute_info_gain, key=total_attribute_info_gain.get)

    def calculate_weighed_entropy(self, entropies, weighs):
        total_entropy = 0.0
        number_of_cases = sum(weighs.values())
        for k, v in weighs.items():
            total_entropy += entropies[k] * (v/number_of_cases)
        return total_entropy

    def change_entropy_to_info_gain(self, entropies):
        entropies_copy = {k: self.main_set_entropy - v for k, v in entropies.items()}
        return entropies_copy

    def create_tree_level(self, partition_attribute):
        if self.level == 1:
            self.tree_root = TreeNode(None, self.level)
        else:
            self.tree_root = TreeNode(self.training_data, self.level)
        self.tree_root.node_name = partition_attribute
        unclassified_nodes_indexes = []
        values_with_nodes = []
        for k, v in self.key_specific_cases_dict[partition_attribute].items():
            child = TreeNode(self.key_specific_cases_dict[partition_attribute][k], self.level+1)
            child.determine_node_name_based_on_case_set()
            child.label_on_branch_before = k
            values_with_nodes.append(k)
            self.tree_root.add_child(child)
            if child.node_name is None:
                unclassified_nodes_indexes.append(self.tree_root.children.index(child))
        # check for ??? nodes (those which have no particular cases in training data
        for element in self.master_set_values[partition_attribute]:
            if element not in values_with_nodes:
                child = TreeNode(None, self.level+1)
                child.label_on_branch_before = element
                self.tree_root.determine_node_name_based_on_case_set()
                child.node_name = self.tree_root.most_frequent_decision
                self.tree_root.add_child(child)

        return unclassified_nodes_indexes

    def trim_node_case_set(self, node_case_set, index):
        node_case_set_copy = []
        for x in node_case_set:
            y = x[:]
            del y[index]
            node_case_set_copy.append(y)
        return node_case_set_copy


class CalculationUtils:
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


class TreeNode:
    """Represents a single node in decision tree

    @:param node_case_set - training cases that are connected with this node
    @:param children - list of this node's children
    @:param label_on_branch_before - label on vertex (branch) that comes from parent of this node to this node
    @:param node_name - label on edge (node)
    @:param level - level of tree on which this node resides
    @:param most_frequent_decision - derived from node_case_set, used in determining node_name of ??? nodes
    """
    def __init__(self, node_case_set, level):
        self.node_case_set = node_case_set
        self.children = []
        self.label_on_branch_before = None
        self.node_name = None
        self.level = level
        self.most_frequent_decision = None

    def add_child(self, child):
        self.children.append(child)

    def determine_node_name_based_on_case_set(self):
        utils = CalculationUtils()
        if self.node_case_set is not None:
            entropy_of_node_case_set = utils.calculate_entropy(self.node_case_set)
            if entropy_of_node_case_set == 0:
                self.node_name = self.node_case_set[0][-1]
            else:
                decisions_dict = utils.create_dict_with_number_of_occurences(self.node_case_set, -1)
                self.most_frequent_decision = max(decisions_dict, key=decisions_dict.get)

    def __str__(self):
        representation = ""
        if self.label_on_branch_before is not None:
            representation += "(" + self.label_on_branch_before + ")"
        if self.node_name is not None:
            representation += " " + self.node_name
        representation += "\n"
        if len(self.children) != 0:
            for child in self.children:
                representation += self.level * "\t" + str(child)
        return representation
