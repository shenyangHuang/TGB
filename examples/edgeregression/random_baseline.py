import numpy as np


from tgb.edgeregression.dataset import EdgeRegressionDataset
from tgb.edgeregression.evaluate import Evaluator

r"""
Show example for data loading and evaluator with a random prediction baseline
"""



def main():

    #process dataset
    name = "un_trade" #"enron"
    enron_tgb = EdgeRegressionDataset(name=name, root="datasets")
    enron_tgb.pre_process()



    # train_data = enron_tgb.train_data    
    # test_data = enron_tgb.test_data
    # val_data = enron_tgb.val_data


    # generate fake data to test the evaluator 
    y_true = np.ones((10,1))
    y_pred = np.zeros((10,1))

    #evaluate performance
    evaluator = Evaluator(name=name)
    print(evaluator.expected_input_format) 
    print(evaluator.expected_output_format) 
    input_dict = {"y_true": y_true, "y_pred": y_pred, 'eval_metric': ['mse']}

    result_dict = evaluator.eval(input_dict) 
    print (result_dict)



if __name__ == "__main__":
    main()