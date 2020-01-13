#ifndef DNN_MODEL_H
#define DNN_MODEL_H

#include <string>
#include <vector>

using namespace std;

class DnnModel
{
	private:
		int									layers;							// The number of DNN hidden layers
		vector< int>						neurons;						// The number of neurons in DNN layers
		int									maximum_neurons;				// Maximum of DNN neurons

		vector< vector< vector<double> > >	weight;							// Floating point DNN weights
		vector< vector<double> >			bias;							// Floating point DNN biases
		vector< double>						input_max;

		vector< vector< vector<double> > >	sfp_weight;						// SFP DNN weights
		vector< vector<double> >			sfp_bias;						// SFP DNN biases
		vector< vector<double> >			sfp_X_hat;						// Normalization factor of neurons outputs
		vector<double>						sfp_output_compensation;		// Quantization factor of DNN weights in output layer

		int									sfp_norm_precision;				// Fraction length of SFP
		double								sfp_two_to_power_of_precision;	// 2^sfp_norm_precision
		double								sfp_max;						// (2^sfp_norm_precision - 1) / 2^sfp_norm_precision
		bool								round_mode;						// 0 for truncation, 1 for rounding to nearest

		double						fix_sfp( double x);						// Convert input value to SFP with specified precision
		void						save_sfp_neuron( string model_name);	// Save SFP DNN model normalized in per neuron

	public:
		DnnModel() {}																			// Construction for no DNN model
		DnnModel( string path_model, string path_dataset);															// Read DNN model
		vector< double>				inference_data( vector< double> dnn_input);					// Read one DNN input data and return DNN outputs
		double						inference_dataset_classification( string path_dataset);		// Read classification DNN input dataset and return accuracy
		vector< vector< double> >	inference_dataset_regression( string path_dataset);			// Read regression DNN input dataset and return DNN output

		void						norm2sfp( int precision, string path_dataset, bool classification, bool round_nearest);	// Normalize in per neuron and classification DNN training dataset
		vector< double>				inference_data_sfp( vector< double> &dnn_input, bool round_nearest);								// Read one DNN input data and inference with SFP DNN normalized in per neuron then return DNN outputs
		double						inference_dataset_classification_sfp( string path_dataset, bool round_nearest);					// Read classification DNN input dataset and inference with SFP DNN normalized in per neuron then return accuracy
		vector< vector< double> >	inference_dataset_regression_sfp( string path_dataset, bool round_nearest);						// Read regression DNN input dataset and inference with SFP DNN normalized in per neuron then return DNN outputs
};

#endif
