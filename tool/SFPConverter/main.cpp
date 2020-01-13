#include <iostream>
#include <cstdlib>
#include <string>

#include "DnnModel.h"

using namespace std;

int main(int argc, char *argv[])
{
	string path_to_model, path_to_training_data, path_to_test_data;
	int fraction_length;

	/* Arguments Parsing */
	string arg;
	for (int i=1; i<argc; i++) {
		arg = argv[i];

		if (arg == "-p") {                  // fraction bit length of fixed point
			fraction_length = atoi(argv[++i]);
		}
		else if (arg == "-m") {             // Path to DNN model
			path_to_model = argv[++i];
		}
		else if (arg == "-td") {            // Path to Training Data
			path_to_training_data = argv[++i];
		}
		else if (arg == "-vd") {            // Path to Test Data
			path_to_test_data = argv[++i];
		}
		else {
			cout<< "Error in argument parsing"<< endl;
			cout<< "Get "<< arg<< endl;
			exit(-1);
		}
	}

	DnnModel model(path_to_model, path_to_training_data);
	cout<< "Base Accuracy: "<< model.inference_dataset_classification(path_to_test_data)<< endl;

	model.norm2sfp(fraction_length, path_to_training_data, true, false);
	cout<< "SFP Accuracy: "<< model.inference_dataset_classification_sfp(path_to_test_data, false)<< endl;

	return 0;
}