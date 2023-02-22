# TGB
Temporal Graph Benchmark project repo 


Data format for CTDG methods (TGAT, TGN, CAWN, and probably any other model that is built upon TGAT):

- The raw data format (ref: https://github.com/srijankr/jodie):
	- The networks are saved as "<network>.csv"
	- The networks are in the following format:
		- one line per interaction/edge
		- Each line is as follows: source, destination, timestamp, state_label, comma-separated array of features
		- First line is the network format
		- if 'source' & 'destination' are alphanumeric, they should be mapped to integers.
		- 'timestamp' should be in cardinal format (not in datetime)
		- 'state_label' should be 1 whenever the source state changes, 0 otherwise. If there are no state labels, use 0 for all interactions. (this label is used for the dynamic node classification task ONLY)
		- 'feature list' can be as long as desired. It should be at least 1 dimensional. If there are no features, use 0 for all interactions.
	- Example:
	source,destination,timestamp,state_label,comma_separated_list_of_features
	0,0,0.0,0,0.1,0.3,10.7
	2,1,6.0,0,0.2,0.4,0.6
	5,0,41.0,0,0.1,15.0,0.6
	3,2,49.0,1,100.7,0.8,0.9

The raw data format is preprocessed to generate three different files that are directly used by the models.
	- 'ml_<network>.csv': source, destination, timestamp, state_label, index 	# 'index' is the index of the line in the edgelist
	- 'ml_<network>.npy': contains the edge features; this is a numpy array where each element corresponds to the features of the corresponding line specifying one edge. If there are no features, should be initialized by zeros.
	- 'ml_<network>_node.npy': contains the node features; this is a numpy array that each element specify the features of one node where the node-id is equal to the element index.
