import numpy as np


from tgb.edgeregression.dataset import EdgeRegressionDataset
from tgb.edgeregression.evaluate import Evaluator

r"""
Show example for data loading and evaluator with a random prediction baseline
"""



def main():

    #process dataset
    name = "un_trade" #"enron"
    dataset = EdgeRegressionDataset(name=name, root="datasets")

    '''
    the following can be obtained from dataset
    '''
    # dataset.node_feat
    # dataset.edge_feat #not the edge weights
    # dataset.full_data
    # dataset.full_data["edge_idxs"]
    # dataset.full_data["sources"]
    # dataset.full_data["destinations"]
    # dataset.full_data["timestamps"] 
    # dataset.full_data["y"]

    # dataset.train_data
    # dataset.val_data
    # dataset.test_data


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