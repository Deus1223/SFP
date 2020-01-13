#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <bitset>

#include "DnnModel.h"

using namespace std;

DnnModel::DnnModel(string path_to_model, string path_dataset)
{
	int read_words;

	/* Open DNN Model File */
	FILE *fp_model = fopen(path_to_model.c_str(), "r");
	if (!fp_model) {
		cout<< "Error - Invalid path to DNN model: "<< path_to_model<< endl;
		exit(-1);
	}

	read_words = fscanf(fp_model, "%d", &this->layers); // layers Number    
	
	this->neurons.resize( this->layers);
	this->weight.resize( this->layers-1);
	this->bias.resize( this->layers-1);
	
	/* Read The Number of Neurons in DNN Layers */
	this->maximum_neurons = 0;
	for (int i=0; i<this->layers; i++) {
		read_words = fscanf(fp_model, "%d", &(this->neurons.at(i)));

		if (this->neurons.at(i) > this->maximum_neurons) {
			this->maximum_neurons = this->neurons.at(i);
		}
	}

	/* Read DNN Parameters */
	long long int hex_input;
	for (int i=1; i < this->layers; i++) {
		this->weight.at(i-1).resize( this->neurons.at(i));
		this->bias.at(i-1).resize( this->neurons.at(i));
		
		for (int j=0; j < this->neurons.at(i); j++) {
			this->weight.at(i-1).at(j).resize(this->neurons.at(i-1));
			
			// Read Weights
			for (int k=0; k < this->neurons.at(i-1); k++) {
				read_words = fscanf(fp_model, "%llx", &hex_input);
				this->weight.at(i-1).at(j).at(k) = *((double*) &hex_input);                
			}
		}

		// Read Biases
		for (int j=0; j < this->neurons.at(i); j++) {
			read_words = fscanf(fp_model, "%llx", &hex_input);
			this->bias.at(i-1).at(j) = *((double*) &hex_input);            
		}
	}

	fclose(fp_model);

	/* Open DNN Input Dataset File */
	int data_num;
	FILE *fp_data = fopen(path_dataset.c_str(), "r");
	if (!fp_data) {
		cout<< "Error - Invalid path to DNN model: "<< path_to_model<< endl;
		exit(-1);
	}
	read_words = fscanf(fp_data, "%d", &data_num);	// Read the number of data

	this->input_max.resize( this->neurons.at(0));
	for (int j=0; j < this->neurons.at(0); j++) {
		this->input_max.at(j) = 1;
	}

	/* Find Out Normalization Factor of DNN Input */
	double input_d;
	int ans;
	for (int n=0; n < data_num; n++) {
		for (int j=0; j < this->neurons.at(0); j++) {
			read_words = fscanf(fp_data, "%llx", &hex_input);
			input_d = *((double*) &hex_input);

			if (abs(input_d) > this->input_max.at(j)) {
				this->input_max.at(j) = abs(input_d);
			}
		}
		read_words = fscanf(fp_data, "%d", &ans);
	}
	fclose(fp_data);
}

vector< double> DnnModel::inference_data( vector< double> dnn_input)
{
	double pp_buffer[2][this->maximum_neurons];		// reused buffer
	int ping=0, pong=1;

	for (int j=0; j < this->neurons.at(0); j++) {
		pp_buffer[ping][j] = dnn_input.at(j);
	}

	/* DNN inference */
	for (int i=1; i < this->layers; i++) {
		for (int j=0; j < this->neurons.at(i); j++) {
			pp_buffer[pong][j] = this->bias.at(i-1).at(j);

			for (int k=0; k < this->neurons.at(i-1); k++) {
				pp_buffer[pong][j] += this->weight.at(i-1).at(j).at(k) * pp_buffer[ping][k];
			}

			// Activation Function
			if (i != this->layers-1) {
				if (pp_buffer[pong][j] < 0) {
					pp_buffer[pong][j] = 0;
				}
			}
		}
		ping = !ping;
		pong = !pong;
	}

	vector< double> dnn_output;
	dnn_output.resize( this->neurons.at( this->layers-1));
	for (int k=0; k < this->neurons[ this->layers - 1]; k++) {
		dnn_output.at(k) = pp_buffer[ping][k];
	}

	return dnn_output;
}

double DnnModel::inference_dataset_classification( string path_dataset)
{
	int read_words;
	
	/* Classification Data to inference */
	FILE *fp_test = fopen(path_dataset.c_str(), "r");
	if (!fp_test) {
		cout<< "Error - Invalid path to test data"<< path_dataset<< endl;
		exit(-1);
	}
	
	int test_num;
	read_words = fscanf(fp_test, "%d", &test_num);	// the number of DNN input data
	
	vector< double> dnn_input, dnn_output;
	dnn_input.resize( this->neurons.at(0));
	dnn_output.resize( this->neurons.at( this->layers-1));
	
	long long int input;
	double result;
	int ans, result_idx;
	int cnt_correct=0;
	for (int t=0; t<test_num; t++) {

		/* Read data and answer */
		for (int i=0; i < this->neurons.at(0); i++) {
			read_words = fscanf(fp_test, "%llx", &input);
			dnn_input.at(i) = *((double*) &input);
		}
		read_words = fscanf(fp_test, "%d", &ans);

		/* DNN inference */
		dnn_output = this->inference_data( dnn_input);

		/* Check answer */
		result = dnn_output.at(0);
		result_idx = 0;
		for (int i=1; i < this->neurons.at( this->layers-1 ); i++) {
			if (dnn_output.at(i) > result) {
				result = dnn_output.at(i);
				result_idx = i;
			}
		}

		if (result_idx == ans) {
			cnt_correct ++;
		}
	}

	fclose(fp_test);

	return (double)cnt_correct / test_num;
}

vector< vector< double> > DnnModel::inference_dataset_regression( string path_dataset)
{
	int read_words;
	
	/* Regression Data to Inference */
	FILE *fp_test = fopen(path_dataset.c_str(), "r");
	if (!fp_test) {
		cout<< "Error - Invalid path to test data"<< path_dataset<< endl;
		exit(-1);
	}
	
	int test_num;
	read_words = fscanf(fp_test, "%d", &test_num);	// the number of DNN input data
	
	vector< double> dnn_input;
	dnn_input.resize( this->neurons.at(0));

	vector< vector< double> > dnn_output;
	dnn_output.resize( test_num);
	for (int t=0; t<test_num; t++) {
		dnn_output.at(t).resize( this->neurons.at( this->layers-1));
	}
	
	long long int input;
	for (int t=0; t<test_num; t++) {

		/* Read data */
		for (int i=0; i < this->neurons.at(0); i++) {
			read_words = fscanf(fp_test, "%llx", &input);
			dnn_input.at(i) = *((double*) &input);
		}

		/* DNN inference for given data */
		dnn_output.at(t) = this->inference_data( dnn_input);
	}

	fclose(fp_test);

	return dnn_output;
}

double DnnModel::fix_sfp(double x)
{
	if (x == 0) {
		return 0;
	}
	else if (x >= 1) {
		return this->sfp_max;
	}
	else if (x <= -1) {
		return -1;
	}
	else {
		double sfp;

		if (this->round_mode) {		// round to nearest
			sfp = round(x * this->sfp_two_to_power_of_precision) / this->sfp_two_to_power_of_precision;

			if (sfp == 1) {
				return this->sfp_max;
			}
			else if (sfp == -1) {
				return -1;
			}
			else {
				return sfp;
			}
		}
		else {						// truncation
			sfp = floor( fabs(x) * (this->sfp_two_to_power_of_precision)) / this->sfp_two_to_power_of_precision;
			
			return (x >= 0) ? sfp : -1*sfp;
		}
	}
}

void DnnModel::norm2sfp( int precision, string path_dataset, bool classification, bool round_nearest)
{
	int read_words;

	this->sfp_weight.resize( this->layers - 1);
	this->sfp_bias.resize( this->layers - 1);
	this->sfp_X_hat.resize( this->layers - 1);
	this->sfp_output_compensation.resize( this->neurons.at( this->layers-1));

	for (int i=1; i < this->layers; i++) {
		this->sfp_weight.at(i-1).resize( this->neurons.at(i));
		this->sfp_bias.at(i-1).resize( this->neurons.at(i));
		this->sfp_X_hat.at(i-1).resize( this->neurons.at(i));

		for (int j=0; j < this->neurons.at(i); j++) {
			this->sfp_weight.at(i-1).at(j).resize(this->neurons.at(i-1));
		}
	}

	this->round_mode = round_nearest;
	this->sfp_norm_precision = precision;
	this->sfp_two_to_power_of_precision = pow(2, precision);
	this->sfp_max = (this->sfp_two_to_power_of_precision - 1) / this->sfp_two_to_power_of_precision;

	for (int j=0; j < this->neurons.at(1); j++) {
		for (int k=0; k < this->neurons.at(0); k++) {
			this->weight.at(0).at(j).at(k) *= this->input_max.at(k);
		}
	}

	/* Data for Normalization */
	FILE *fp_test = fopen(path_dataset.c_str(), "r");
	if (!fp_test) {
		cout<< "Error - Invalid path to test data: "<< path_dataset<< endl;
		exit(-1);
	}
	
	int test_num;
	read_words = fscanf(fp_test, "%d", &test_num);  // the number of DNN input data

	double last_x[this->maximum_neurons], now_x[this->maximum_neurons],
			now_w[this->maximum_neurons][this->maximum_neurons], now_b[this->maximum_neurons],
			max_w[this->layers-1][this->maximum_neurons];

	/* Initialize what and Xhat */
	for (int i=0; i < this->layers-1; i++) {
		for (int j=0; j < this->maximum_neurons; j++) {
			max_w[i][j] = 0;
		}
		for (int j=0; j < this->neurons.at(i+1); j++) {
			this->sfp_X_hat.at(i).at(j) = 1;
		}
	}
	
	cout<< "Start normalizing in per neuron and input data"<< endl;
	
	int ans;
	long long int input;
	for (int t=0; t<test_num; t++) {
		for (int i=0; i < this->neurons.at(0); i++) {
			read_words = fscanf(fp_test, "%llx", &input);
			last_x[i] = *((double *) &input);
			last_x[i] /= this->input_max.at(i);
			last_x[i] = this->fix_sfp( last_x[i]);      // Input data must also be SFP
		}

		if (classification) {
			read_words = fscanf(fp_test, "%d", &ans);
		}

		/* Hidden Layer 1 */
		for (int j=0; j < this->neurons.at(1); j++) {
			
			/* Update what */
			if (abs(this->bias.at(0).at(j)) > max_w[0][j]) {
				max_w[0][j] = abs(this->bias.at(0).at(j));
			}
			for (int k=0; k < this->neurons.at(0); k++) {
				if (abs(this->weight.at(0).at(j).at(k)) > max_w[0][j]) {
					max_w[0][j] = abs(this->weight.at(0).at(j).at(k));
				}
			}

			/* Normalize hidden layer 1 weights and biases with present what */
			now_b[j] = this->bias.at(0).at(j) / max_w[0][j];
			now_b[j] = this->fix_sfp( now_b[j]);
			for (int k=0; k < this->neurons.at(0); k++) {
				now_w[j][k] = this->weight.at(0).at(j).at(k) / max_w[0][j];
				now_w[j][k] = this->fix_sfp( now_w[j][k]);
			}
		}
		for (int j=0; j < this->neurons.at(1); j++) {   
			
			/* Update Xhat */
			now_x[j] = now_b[j];
			for (int k=0; k < this->neurons.at(0); k++) {
				now_x[j] += now_w[j][k] * last_x[k];
			}

			if (now_x[j] >= this->sfp_X_hat.at(0).at(j)) {
				this->sfp_X_hat.at(0).at(j) = pow(2, ceil(log2(now_x[j])));

				/* Prevent normalized neuron output to be 1 */
				if (this->sfp_X_hat.at(0).at(j) == now_x[j]) {
					this->sfp_X_hat.at(0).at(j) *= 2;
				}
			}
		}
		for (int j=0; j < this->neurons.at(1); j++) {
			
			/* Generate neurons output for next layer */
			if (now_x[j] > 0) {
				last_x[j] = now_x[j] / this->sfp_X_hat.at(0).at(j);
				last_x[j] = this->fix_sfp( last_x[j]);
			}
			else {
				last_x[j] = 0;
			}
		}

		/* Hidden Layer 2 ~ Output Layer */
		for (int i=2; i < this->layers; i++) {
			for (int j=0; j < this->neurons.at(i); j++) {
				
				/* Update What */
				if (abs(this->bias.at(i-1).at(j)) > max_w[i-1][j]) {
					max_w[i-1][j] = abs(this->bias.at(i-1).at(j));
				}
				for (int k=0; k < this->neurons.at(i-1); k++) {
					now_w[j][k] = this->weight.at(i-1).at(j).at(k) * max_w[i-2][k] * this->sfp_X_hat.at(i-2).at(k); // weights compensation
					if (abs(now_w[j][k]) > max_w[i-1][j]) {
						max_w[i-1][j] = abs(now_w[j][k]);
					}
				}

				/* Normalize weights and biases with present what */
				now_b[j] = this->bias.at(i-1).at(j) / max_w[i-1][j];
				now_b[j] = this->fix_sfp( now_b[j]);
				for (int k=0; k < this->neurons.at(i-1); k++) {
					now_w[j][k] = now_w[j][k] / max_w[i-1][j];
					now_w[j][k] = this->fix_sfp( now_w[j][k]);
				}
			}
			for (int j=0; j < this->neurons.at(i); j++) {
				
				/* Update Xhat */
				now_x[j] = now_b[j];
				for (int k=0; k < this->neurons.at(i-1); k++) {
					now_x[j] += now_w[j][k] * last_x[k];
				}

				if (i != this->layers-1) {
					if (now_x[j] >= this->sfp_X_hat.at(i-1).at(j)) {
						this->sfp_X_hat.at(i-1).at(j) = pow(2, ceil(log2(now_x[j])));
						
						/* Prevent normalized neuron output to be 1 */
						if (this->sfp_X_hat.at(i-1).at(j) == now_x[j]) {
							this->sfp_X_hat.at(i-1).at(j) *= 2;
						}
					}
				}
				else {
					if (abs(now_x[j]) >= this->sfp_X_hat.at(i-1).at(j)) {
						this->sfp_X_hat.at(i-1).at(j) = pow(2, ceil(log2(abs(now_x[j]))));
						
						/* Prevent normalized neuron output to be 1 */
						if (this->sfp_X_hat.at(i-1).at(j) == abs(now_x[j])) {
							this->sfp_X_hat.at(i-1).at(j) *= 2;
						}
					}
				}
			}
			for (int j=0; j < this->neurons.at(i); j++) {

				/* Generate neurons output for next layer */
				if (now_x[j] > 0) {
					last_x[j] = now_x[j] / this->sfp_X_hat.at(i-1).at(j);
					last_x[j] = this->fix_sfp( last_x[j]);
				}
				else {
					last_x[j] = 0;
				}
			}
		}
		
		if((t+1)%1000 == 0){
			cout<< "Normalizing in per neuron and input data ... "<< t+1<< "/"<< test_num<< endl;
		}
	}

	/* Normalize Weights and Biases */
	for (int j=0; j < this->neurons.at(1); j++) {
		this->sfp_bias.at(0).at(j) = this->bias.at(0).at(j) / max_w[0][j];
		this->sfp_bias.at(0).at(j) = this->fix_sfp( this->sfp_bias.at(0).at(j));

		for (int k=0; k < this->neurons.at(0); k++) {
			this->sfp_weight.at(0).at(j).at(k) = this->weight.at(0).at(j).at(k) / max_w[0][j];
			this->sfp_weight.at(0).at(j).at(k) = this->fix_sfp( this->sfp_weight.at(0).at(j).at(k));
		}
	}
	for (int i=2; i < this->layers; i++) {
		for (int j=0; j < this->neurons.at(i); j++) {
			this->sfp_bias.at(i-1).at(j) = this->bias.at(i-1).at(j) / max_w[i-1][j];
			this->sfp_bias.at(i-1).at(j) = this->fix_sfp( this->sfp_bias.at(i-1).at(j));

			for (int k=0; k < this->neurons.at(i-1); k++) {
				this->sfp_weight.at(i-1).at(j).at(k) = (this->weight.at(i-1).at(j).at(k) * max_w[i-2][k] * this->sfp_X_hat.at(i-2).at(k)) / max_w[i-1][j];
				this->sfp_weight.at(i-1).at(j).at(k) = this->fix_sfp( this->sfp_weight.at(i-1).at(j).at(k));
			}
		}
	}

	for (int j=0; j < this->neurons.at(this->layers-1); j++) {
		this->sfp_output_compensation.at(j) = max_w[this->layers-2][j] * this->sfp_X_hat.at( this->layers-2 ).at(j);
	}

	this->save_sfp_neuron("model_acc");
}

void DnnModel::save_sfp_neuron( string model_name)
{
	char outputFile[50];
	sprintf(outputFile, "%s_%d.txt", (string("./model/") + model_name).c_str(), this->sfp_norm_precision);
	FILE *fp_model = fopen(outputFile, "w");
	if (!fp_model) {
		cout<< "Can't save normalized model in norm_neuron"<< endl;
		exit(-1);
	}

	short output;
	bitset<16> output_b;
	for (int i=1; i < this->layers; i++) {
		for (int j=0; j < this->neurons.at(i); j++) {
			output = log2( this->sfp_X_hat.at(i-1).at(j));		// output shift
			output_b = output;
			fprintf(fp_model, "%s", output_b.to_string().c_str());

			output = this->sfp_bias.at(i-1).at(j) * this->sfp_two_to_power_of_precision;	// bias
			output_b = output;
			fprintf(fp_model, "%s\n", output_b.to_string().c_str());

			for (int k=0; k < this->neurons.at(i-1); k+=2) {	// weights
				if ((k+1) < this->neurons.at(i-1)) {
					output = this->sfp_weight.at(i-1).at(j).at(k+1) * this->sfp_two_to_power_of_precision;
					output_b = output;
					fprintf(fp_model, "%s", output_b.to_string().c_str());
				}
				else {
					output_b = 0;
					fprintf(fp_model, "%s", output_b.to_string().c_str());
				}

				output = this->sfp_weight.at(i-1).at(j).at(k) * this->sfp_two_to_power_of_precision;
				output_b = output;
				fprintf(fp_model, "%s\n", output_b.to_string().c_str());
			}
		}
	}
	fclose(fp_model);

	FILE *fp_comp = fopen("./model/SFP_HEAD.h", "w");
	if (!fp_comp) {
		cout<< "Can't save SFP_HEAD in norm_neuron"<< endl;
		exit(-1);
	}
	
	fprintf(fp_comp, "#ifndef SFP_HEAD_H\n");
	fprintf(fp_comp, "#define SFP_HEAD_H\n\n");

	fprintf(fp_comp, "float input_Xhat[] = {\n");
	for (int j=0; j < this->neurons.at(0); j++) {
		fprintf(fp_comp, "\t%f", this->input_max.at(j));

		if ((j+1) < this->neurons.at(0)) {
			fprintf(fp_comp, ",");
		}
		fprintf(fp_comp, "\n");
	}
	fprintf(fp_comp, "};\n");

	fprintf(fp_comp, "float compensation[] = {\n");
	for (int j=0; j < this->neurons.at( this->layers-1); j++) {
		fprintf(fp_comp, "\t%f", this->sfp_output_compensation.at(j));

		if ((j+1) < this->neurons.at( this->layers-1)) {
			fprintf(fp_comp, ",");
		}
		fprintf(fp_comp, "\n");
	}
	fprintf(fp_comp, "};\n");
	
	fprintf(fp_comp, "#endif");
	fclose(fp_comp);

	FILE *fp_config = fopen("./model/config.txt", "w");
	fprintf(fp_config, "%d\n", this->layers - 2);
	for (int i=0; i < this->layers; i++) {
		fprintf(fp_config, "%d\n", this->neurons.at(i) - 1);
	}
	fclose(fp_config);
}

vector< double> DnnModel::inference_data_sfp( vector< double> &dnn_input, bool round_nearest)
{
	this->round_mode = round_nearest;
	
	double pp_buffer[2][this->maximum_neurons];					// reused Buffer
	int ping=0, pong=1;

	for (int i=0; i < this->neurons.at(0); i++) {
		dnn_input.at(i) = this->fix_sfp( dnn_input.at(i));
		pp_buffer[ping][i] = dnn_input.at(i);	// input data must also be SFP
	}

	for (int i=1; i < this->layers; i++) {
		for (int j=0; j < this->neurons.at(i); j++) {
			pp_buffer[pong][j] = this->sfp_bias.at(i-1).at(j);

			for (int k=0; k < this->neurons.at(i-1); k++) {
				pp_buffer[pong][j] += this->sfp_weight.at(i-1).at(j).at(k) * pp_buffer[ping][k];
			}

			/* ReLU then normalize and convert to SFP */
			if (i != this->layers-1) {
				if (pp_buffer[pong][j] <= 0) {
					pp_buffer[pong][j] = 0;
				}
				else {
					pp_buffer[pong][j] /= this->sfp_X_hat.at(i-1).at(j);    // Normalization First
					pp_buffer[pong][j] = this->fix_sfp( pp_buffer[pong][j]);
				}
			}
			else {
				pp_buffer[pong][j] /= this->sfp_X_hat.at(i-1).at(j);        // Normalization First
				pp_buffer[pong][j] = this->fix_sfp( pp_buffer[pong][j]);
			}
		}
		ping = !ping;
		pong = !pong;
	}

	vector< double> dnn_output;
	dnn_output.resize( this->neurons.at( this->layers-1));
	for (int k=0; k < this->neurons[ this->layers - 1]; k++) {
		dnn_output.at(k) = pp_buffer[ping][k];
	}

	return dnn_output;
}

double DnnModel::inference_dataset_classification_sfp( string path_dataset, bool round_nearest)
{
	int read_words;

	/* Test Data */
	FILE *fp_test = fopen(path_dataset.c_str(), "r");
	if (!fp_test) {
		cout<< "Error - Invalid path to test data"<< path_dataset<< endl;
		exit(-1);
	}

	int data_num;
	read_words = fscanf(fp_test, "%d", &data_num);  // Test data amount

	vector< double> dnn_input, dnn_output;
	dnn_input.resize( this->neurons.at(0));
	dnn_output.resize( this->neurons.at( this->layers-1));

	long long int input;
	double result;
	int ans, result_idx;
	int cnt_correct=0;

	FILE *fp_dnnin = fopen("./model/dnnin.txt", "w");
	if (!fp_dnnin) {
		cout<< "Failed to DNN Input file"<< endl;
		exit(-1);
	}

	FILE *fp_dnnout = fopen("./model/dnnout.txt", "w");
	if (!fp_dnnout) {
		cout<< "Failed to DNN Input file"<< endl;
		exit(-1);
	}

	short output;
	bitset<16> output_b;
	for (int t=0; t<data_num; t++) {
		for (int j=0; j < this->neurons.at(0); j++) {
			read_words = fscanf(fp_test, "%llx", &input);
			dnn_input.at(j) = *((double*) &input);
			dnn_input.at(j) /= this->input_max.at(j);
		}
		read_words = fscanf(fp_test, "%d", &ans);

		dnn_output = this->inference_data_sfp( dnn_input, round_nearest);

		for (int j=0; j < this->neurons.at(0); j+=2) {
			if ((j+1) < this->neurons.at(0)) {
				output = dnn_input.at(j+1) * this->sfp_two_to_power_of_precision;
				output_b = output;
				fprintf(fp_dnnin, "%s", output_b.to_string().c_str());
			}
			else {
				output_b = 0;
				fprintf(fp_dnnin, "%s", output_b.to_string().c_str());
			}

			output = dnn_input.at(j) * this->sfp_two_to_power_of_precision;
			output_b = output;
			fprintf(fp_dnnin, "%s\n", output_b.to_string().c_str());
		}
		for (int j=0; j < this->neurons.at( this->layers-1 ); j+=2) {
			if ((j+1) < this->neurons.at( this->layers-1 )) {
				output = floor(dnn_output.at(j+1) * this->sfp_two_to_power_of_precision);
				fprintf(fp_dnnout, "%04hx", output);
			}
			else {
				fprintf(fp_dnnout, "%04hx", 0);
			}

			output = floor(dnn_output.at(j) * this->sfp_two_to_power_of_precision);
			fprintf(fp_dnnout, "%04hx\n", output);
		}
		
		/* Compensation */
		for (int j=0; j < this->neurons.at( this->layers-1 ); j++) {
			dnn_output.at(j) = dnn_output.at(j) * this->sfp_output_compensation.at(j);
		}

		/* Check Answer */
		result = dnn_output.at(0);
		result_idx = 0;
		for (int j=1; j < this->neurons.at( this->layers-1 ); j++) {
			if (dnn_output.at(j) > result) {
				result = dnn_output.at(j);
				result_idx = j;
			}
		}

		if (result_idx == ans) {
			cnt_correct ++;
		}
	}

	fclose(fp_dnnin);
	fclose(fp_dnnout);
	fclose(fp_test);

	return (double)cnt_correct / data_num;
}

vector< vector< double> > DnnModel::inference_dataset_regression_sfp( string path_dataset, bool round_nearest)
{
	int read_words;

	/* Test Data */
	FILE *fp_test = fopen(path_dataset.c_str(), "r");
	if (!fp_test) {
		cout<< "Error - Invalid path to test data"<< path_dataset<< endl;
		exit(-1);
	}

	int data_num;
	read_words = fscanf(fp_test, "%d", &data_num);  // Test data amount

	vector< double> dnn_input;
	dnn_input.resize( this->neurons.at(0));
	
	vector< vector< double> > dnn_output;
	dnn_output.resize( data_num);
	for (int t=0; t<data_num; t++) {
		dnn_output.at(t).resize( this->neurons.at( this->layers-1));
	}

	FILE *fp_dnnin = fopen("./DnnIO/dnnin.txt", "w");
	if (!fp_dnnin) {
		cout<< "Failed to DNN Input file"<< endl;
		exit(-1);
	}

	FILE *fp_dnnout = fopen("./DnnIO/dnnout.txt", "w");
	if (!fp_dnnout) {
		cout<< "Failed to DNN Input file"<< endl;
		exit(-1);
	}

	/* DNN Inference for regression dataset */
	long long int input;
	short output;
	bitset<16> output_b;
	for (int t=0; t<data_num; t++) {

		/* Read DNN input data */
		for (int j=0; j < this->neurons.at(0); j++) {
			read_words = fscanf(fp_test, "%llx", &input);
			dnn_input.at(j) = *((double*) &input);
			dnn_input.at(j) /= this->input_max.at(j);
		}

		/* DNN inference for given data */
		dnn_output.at(t) = this->inference_data_sfp( dnn_input, round_nearest);

		for (int j=0; j < this->neurons.at(0); j+=2) {
			if ((j+1) < this->neurons.at(0)) {
				output = dnn_input.at(j+1) * this->sfp_two_to_power_of_precision;
				output_b = output;
				fprintf(fp_dnnin, "%s", output_b.to_string().c_str());
			}
			else {
				output_b = 0;
				fprintf(fp_dnnin, "%s", output_b.to_string().c_str());
			}

			output = dnn_input.at(j) * this->sfp_two_to_power_of_precision;
			output_b = output;
			fprintf(fp_dnnin, "%s\n", output_b.to_string().c_str());
		}
		for (int j=0; j < this->neurons.at( this->layers-1 ); j+=2) {
			if ((j+1) < this->neurons.at( this->layers-1 )) {
				output = dnn_output.at(t).at(j+1) * this->sfp_two_to_power_of_precision;
				fprintf(fp_dnnout, "%04hx", output);
			}
			else {
				output_b = 0;
				fprintf(fp_dnnout, "%04hx", 0);
			}

			output = dnn_output.at(t).at(j) * this->sfp_two_to_power_of_precision;
			fprintf(fp_dnnout, "%04hx\n", output);
		}

		/* Compensation */
		for (int j=0; j < this->neurons.at( this->layers-1 ); j++) {
			dnn_output.at(t).at(j) = dnn_output.at(t).at(j) * this->sfp_output_compensation.at(j);
		}
	}

	fclose(fp_dnnin);
	fclose(fp_dnnout);
	fclose(fp_test);

	return dnn_output;
}