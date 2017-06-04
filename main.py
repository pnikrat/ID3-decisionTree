import sys
from case_runner import SingleAlgorithmCase


def main():
    """
    How to run:
    $ python main.py grades.csv
    Training data should be in a file put as the first argument. First line of training data are attribute labels.
    """
    if check_for_correct_number_of_cli_arguments():
        runned_case = SingleAlgorithmCase(sys.argv[1])
        runned_case.start()
    else:
        return


def check_for_correct_number_of_cli_arguments():
    """Return True if user provides exactly 1 argument"""
    if len(sys.argv) == 2:
        return True
    elif len(sys.argv) > 2:
        print("Incorrect number of arguments")
    else:
        print("Please provide csv filename with training data as argument")
    return False


if __name__ == '__main__':
    main()
