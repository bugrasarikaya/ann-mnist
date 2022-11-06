#include <iostream> //This program was developed in Visual Studio 2019.
#include <conio.h>
using namespace std;
int *count_rc_cc_mrl(char* directory) { //Counts lines and columns, calculates max string length in a given csv file to determine array's size for determining dynamic allocation.
	FILE* fp;
	fopen_s(&fp, directory, "r");
	int row_count, column_count, max_row_length, length, chr_count;
	row_count = column_count = max_row_length = length = 0;
	if (fp == NULL) perror("ERROR");
	else {
		char chr = getc(fp);
		do {
			chr_count = 1;
			while (chr != '\n') {
				if (chr == ',' && row_count == 0) column_count++;
				chr = getc(fp);
				chr_count++;
			}
			if (row_count == 0)column_count++; //Only in first row, it counts columnt count because other rows have same column count.
			row_count++;
			length = chr_count - 1;
			if (length > max_row_length) max_row_length = length; //Calculates max row length.
			chr = getc(fp);
		} while (chr != EOF);
		fclose(fp);
	}
	int* temp_array = new int [3];
	temp_array[0] = row_count;
	temp_array[1] = column_count;
	temp_array[2] = max_row_length;
	return temp_array;
}
void read_csv(char* directory, int max_length, double** csv_array) { //Reads csv file and inserts into a 2D integer array.
	FILE* fp;
	fopen_s(&fp, directory, "r");
	if (fp == NULL) perror("ERROR");
	else {
		char* line = new char[max_length + 2]; //Implements dynamic allocation in a sufficient size which is the result of "string_length + 1('\n') + 1('\0')".
		char* token = NULL;
		char* context = NULL;
		int column;
		int row = -1;
		while (fgets(line, (max_length + 2), fp) != NULL) {
			if (row == -1) {
				row++;
				continue;
			}
			token = strtok_s(line, ",", &context); //Seperates line by commas.
			column = 0;
			do {
				csv_array[row][column++] = atof(token);
				token = strtok_s(NULL, ",", &context);
			} while (token != NULL);
			row++;
		}
		delete[] line;
		fclose(fp);
	}
}
struct input_node { //Data of a node.
	double* out_weights;
	double pixel_value;
	double* error_out_weights;
};
struct hidden_node { //Data of a node.
	double* out_weights;
	double activation;
	double bias;
	double sigma;
	double* error_out_weights;
};
struct output_node { //Data of a node.
	double output_activation;
	double bias;
	double sigma;
	double error;
};
struct input_layer{ //Data of an input layer.
	int nodes_number;
	struct input_node* nodes;
};
struct hidden_layer { //Data of a hidden layer.
	int nodes_number;
	struct hidden_node* nodes;
};
struct output_layer { //Data of an output layer.
	int nodes_number;
	struct output_node* nodes;
};
struct input_node* create_input_node(int pixels_number, int next_nodes_number) { //Crates an input layer node array with gvien pixels' number.
	struct input_node* temp_input_nodes = new struct input_node[pixels_number];
	for (int i = 0; i < pixels_number; i++)temp_input_nodes[i].out_weights = new double[next_nodes_number];
	for (int i = 0; i < pixels_number; i++)temp_input_nodes[i].error_out_weights = new double[next_nodes_number];
	return temp_input_nodes;
};
struct hidden_node* create_hidden_node(int nodes_number, int next_nodes_number) { //Crates a hidden layer node array with gvien nodes' number.
	struct hidden_node* temp_hidden_nodes = new struct hidden_node[nodes_number];
	for (int i = 0; i < nodes_number; i++)temp_hidden_nodes[i].out_weights = new double[next_nodes_number];
	for (int i = 0; i < nodes_number; i++)temp_hidden_nodes[i].error_out_weights = new double[next_nodes_number];
	return temp_hidden_nodes;
};
struct output_node* create_output_node(int nodes_number) { //Crates an output layer node array with gvien nodes' number.
	struct output_node* temp_output_nodes = new struct output_node[nodes_number];
	return temp_output_nodes;
};
struct input_layer* create_input_layer(int pixels_number, int next_nodes_number) { //Crates an input layer with gvien pixels' number.
	struct input_layer* temp_input_layer = new struct input_layer();
	(*temp_input_layer).nodes_number = pixels_number;
	(*temp_input_layer).nodes = create_input_node(pixels_number, next_nodes_number);
	return temp_input_layer;
};
struct hidden_layer* create_hidden_layer(int nodes_number, int next_nodes_number) { //Crates a hidden layer with gvien nodes' number.
	struct hidden_layer* temp_hidden_layer = new struct hidden_layer();
	(*temp_hidden_layer).nodes_number = nodes_number;
	(*temp_hidden_layer).nodes = create_hidden_node(nodes_number, next_nodes_number);
	return temp_hidden_layer;
};
struct output_layer* create_output_layer(int nodes_number) { //Crates an output layer with gvien nodes' number.
	struct output_layer* temp_output_layer = new struct output_layer();
	(*temp_output_layer).nodes_number = nodes_number;
	(*temp_output_layer).nodes = create_output_node(nodes_number);
	return temp_output_layer;
};
double sigmoid(double sigma) { //Implements activation function of sigmoid.
	return 1 / 1 + exp(-sigma);
}
double derivative_sigmoid(double output) { //Implements dericative of sigmoid.
	return output * (1 - output);
}
double ReLU(double sigma) { //Implements activation function of ReLU.
	if (sigma > 0) return sigma;
	else return 0;
}
double derivative_ReLU(double output) { //Implements dericative of ReLU.
	if (output > 0) return 1;
	else return 0;
}
double TanH(double sigma) { //Implements activation function of TanH.
	return (2 / (1 + exp(-2 * sigma))) - 1;
}
double derivative_TanH(double output) { //Implements dericative of TanH.
	return 1 - pow(output, 2);
}
void initialize_input_node(int next_nodes_number, struct input_node* nodes, int node_index) {
	for (int i = 0; i < next_nodes_number; i++) {
		(nodes[node_index]).out_weights[i] = ((double)rand()) / ((double)RAND_MAX);
	}
	(nodes[node_index]).pixel_value = 0;
	for (int i = 0; i < next_nodes_number; i++) {
		(nodes[node_index]).error_out_weights[i] = 0;
	}
}
void initialize_hidden_node(int next_nodes_number, struct hidden_node* nodes, int node_index) {
	for (int i = 0; i < next_nodes_number; i++) {
		(nodes[node_index]).out_weights[i] = ((double)rand()) / ((double)RAND_MAX);
	}
	(nodes[node_index]).activation = 0;
	(nodes[node_index]).bias = ((double)rand()) / ((double)RAND_MAX);
	(nodes[node_index]).sigma = 0;
	for (int i = 0; i < next_nodes_number; i++) {
		(*nodes).error_out_weights[i] = 0;
	}
}
void initialize_output_node(struct output_node* nodes, int node_index) {
	(nodes[node_index]).output_activation = 0;
	(nodes[node_index]).bias = ((double)rand()) / ((double)RAND_MAX);
	(nodes[node_index]).sigma = 0;
	(nodes[node_index]).error = 0;
}
void initialize_ann(int* rc_cc_mrl_array, struct input_layer** input_layer, struct hidden_layer** hidden_layer_1, struct hidden_layer** hidden_layer_2, struct output_layer** output_layer, int* next_nodes_number_array) {
	for (int i = 0; i < rc_cc_mrl_array[1] - 1; i++) initialize_input_node(next_nodes_number_array[0], (**input_layer).nodes, i);
	for (int i = 0; i < next_nodes_number_array[0]; i++) initialize_hidden_node(next_nodes_number_array[1], (**hidden_layer_1).nodes, i);
	for (int i = 0; i < next_nodes_number_array[1]; i++) initialize_hidden_node(next_nodes_number_array[2], (**hidden_layer_2).nodes, i);
	for (int i = 0; i < next_nodes_number_array[2]; i++) initialize_output_node((**output_layer).nodes, i );
}
void insert_pixels_values(struct input_layer input_layer, double** csv_array, int example_number) {
	for (int i = 0; i < input_layer.nodes_number; i++) input_layer.nodes[i].pixel_value = csv_array[example_number][i + 1];
}
void forward_propogation(struct input_layer* input_layer, struct hidden_layer* hidden_layer_1, struct hidden_layer* hidden_layer_2, struct output_layer* output_layer, double (*activation_function)(double)) {
	for (int i = 0; i < (*hidden_layer_1).nodes_number; i++){ //Progression of "hidden_layer_1".
		for (int j = 0; j < (*input_layer).nodes_number; j++) {
			(*hidden_layer_1).nodes[i].sigma += (*input_layer).nodes[j].pixel_value * (*input_layer).nodes[j].out_weights[i];
		}
		(*hidden_layer_1).nodes[i].sigma += (*hidden_layer_1).nodes[i].bias;
		(*hidden_layer_1).nodes[i].activation = activation_function((*hidden_layer_1).nodes[i].sigma);
	}
	for (int i = 0; i < (*hidden_layer_2).nodes_number; i++) { //Progression of "hidden_layer_2".
		for (int j = 0; j < (*hidden_layer_1).nodes_number; j++) {
			(*hidden_layer_2).nodes[i].sigma += (*hidden_layer_1).nodes[j].activation * (*hidden_layer_1).nodes[j].out_weights[i];
		}
		(*hidden_layer_2).nodes[i].sigma += (*hidden_layer_2).nodes[i].bias;
		(*hidden_layer_2).nodes[i].activation = activation_function((*hidden_layer_2).nodes[i].sigma);
	}
	for (int i = 0; i < (*output_layer).nodes_number; i++) { //Progression of "output_layer".
		for (int j = 0; j < (*hidden_layer_2).nodes_number; j++) {
			(*output_layer).nodes[i].sigma += (*hidden_layer_2).nodes[j].activation * (*hidden_layer_2).nodes[j].out_weights[i];
		}
		(*output_layer).nodes[i].sigma += (*output_layer).nodes[i].bias;
		(*output_layer).nodes[i].output_activation = activation_function((*output_layer).nodes[i].sigma);
		//cout << i << ": " << (*output_layer).nodes[i].output_activation << endl;
	}
}
void back_propogation(struct input_layer* input_layer, struct hidden_layer* hidden_layer_1, struct hidden_layer* hidden_layer_2, struct output_layer* output_layer, double** csv_array, int example_number, double (*derivative_activation_function)(double)) {
	for (int i = 0; i < (*output_layer).nodes_number; i++) { //Progression of "output_layer".
		(*output_layer).nodes[i].error = (csv_array[example_number][0] - (*output_layer).nodes[i].output_activation) * derivative_activation_function((*output_layer).nodes[i].output_activation);
	}
	for (int i = 0; i < (*hidden_layer_2).nodes_number; i++) { //Progression of "hidden_layer_2".
		for (int j = 0; j < (*output_layer).nodes_number; j++) {
			(*hidden_layer_2).nodes[i].error_out_weights[j] = (*hidden_layer_2).nodes[i].out_weights[j] * (*output_layer).nodes[j].error * derivative_activation_function((*hidden_layer_2).nodes[i].activation);
		}
	}
	for (int i = 0; i < (*hidden_layer_1).nodes_number; i++) { //Progression of "hidden_layer_1".
		for (int j = 0; j < (*hidden_layer_2).nodes_number; j++) {
			for (int k = 0; k < (*output_layer).nodes_number; k++) {
				(*hidden_layer_1).nodes[i].error_out_weights[j] += (*hidden_layer_1).nodes[i].out_weights[j] * (*hidden_layer_2).nodes[j].error_out_weights[k];
			}
			(*hidden_layer_1).nodes[i].error_out_weights[j] *= derivative_activation_function((*hidden_layer_1).nodes[i].activation);
		}
	}
	for (int i = 0; i < (*input_layer).nodes_number; i++) { //Progression of "input_layer".
		for (int j = 0; j < (*hidden_layer_1).nodes_number; j++) {
			for (int k = 0; k < (*hidden_layer_2).nodes_number; k++) {
				(*input_layer).nodes[i].error_out_weights[j] += (*input_layer).nodes[i].out_weights[j] * (*hidden_layer_1).nodes[j].error_out_weights[k];
			}
		}
	}
}
void update(struct input_layer* input_layer, struct hidden_layer* hidden_layer_1, struct hidden_layer* hidden_layer_2, struct output_layer* output_layer, double learning_rate) {
	for (int i = 0; i < (*input_layer).nodes_number; i++) { //Progression of "input_layer".
		for (int j = 0; j < (*hidden_layer_1).nodes_number; j++) (*input_layer).nodes[i].out_weights[j] -= learning_rate * (*input_layer).nodes[i].error_out_weights[j] * (*input_layer).nodes[i].pixel_value;
	}
	for (int i = 0; i < (*hidden_layer_1).nodes_number; i++) { //Progression of "hidden_layer_1".
		for (int j = 0; j < (*hidden_layer_2).nodes_number; j++) (*hidden_layer_1).nodes[i].out_weights[j] -= learning_rate * (*hidden_layer_1).nodes[i].error_out_weights[j] * (*hidden_layer_1).nodes[i].activation;
		for (int j = 0; j < (*hidden_layer_2).nodes_number; j++) (*hidden_layer_1).nodes[i].bias -= learning_rate * (*hidden_layer_1).nodes[i].out_weights[j];
	}
	for (int i = 0; i < (*hidden_layer_2).nodes_number; i++) { //Progression of "hidden_layer_2".
		for (int j = 0; j < (*output_layer).nodes_number; j++) (*hidden_layer_2).nodes[i].out_weights[j] -= learning_rate * (*hidden_layer_2).nodes[i].error_out_weights[j] * (*hidden_layer_2).nodes[i].activation;
		for (int j = 0; j < (*output_layer).nodes_number; j++) (*hidden_layer_2).nodes[i].bias -= learning_rate * (*hidden_layer_2).nodes[i].out_weights[j];
	}
	for (int i = 0; i < (*output_layer).nodes_number; i++) { //Progression of "output_layer".
		(*output_layer).nodes[i].bias -= learning_rate * (*output_layer).nodes[i].error;
	}
}
void train_ann(struct input_layer** input_layer, struct hidden_layer** hidden_layer_1, struct hidden_layer** hidden_layer_2, struct output_layer** output_layer, double** csv_array, int examples_number, double (*activation_function)(double), double (*derivative_activation_function)(double), double learning_rate) {
	for (int i = 0; i < examples_number; i++) {
		insert_pixels_values(**input_layer, csv_array, i);
		forward_propogation(*input_layer, *hidden_layer_1, *hidden_layer_2, *output_layer, *activation_function);
		//cout << (**hidden_layer_1).nodes[0].activation << endl;
		back_propogation(*input_layer, *hidden_layer_1, *hidden_layer_2, *output_layer, csv_array, i, *derivative_activation_function);
		update(*input_layer, *hidden_layer_1, *hidden_layer_2, *output_layer, learning_rate);
	}
}
double accuracy(struct output_layer output_layer, double** csv_array, int examples_number) { //Calculates accuracy.
	double correctness = 0;
	for (int i = 0; i < examples_number; i++) {
		if (csv_array[i][0] == 0) {
			if (output_layer.nodes[0].output_activation == 1) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 1) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 1) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 2) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 1) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 3) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 1) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 4) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 1) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 5) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 1) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 6) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 1) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 7) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 1) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 8) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 1) {
												if (output_layer.nodes[9].output_activation == 0) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else if (csv_array[i][0] == 9) {
			if (output_layer.nodes[0].output_activation == 0) {
				if (output_layer.nodes[1].output_activation == 0) {
					if (output_layer.nodes[2].output_activation == 0) {
						if (output_layer.nodes[3].output_activation == 0) {
							if (output_layer.nodes[4].output_activation == 0) {
								if (output_layer.nodes[5].output_activation == 0) {
									if (output_layer.nodes[6].output_activation == 0) {
										if (output_layer.nodes[7].output_activation == 0) {
											if (output_layer.nodes[8].output_activation == 0) {
												if (output_layer.nodes[9].output_activation == 1) {
													correctness++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	double accuracy_percent = (correctness * 100) / examples_number; //Calculates percent value.
	return accuracy_percent;
}
void test_ann(struct input_layer** input_layer, struct hidden_layer** hidden_layer_1, struct hidden_layer** hidden_layer_2, struct output_layer** output_layer, double** csv_array, int examples_number, double (*activation_function)(double)) {
	for (int i = 0; i < examples_number; i++) {
		insert_pixels_values(**input_layer, csv_array, i);
		forward_propogation(*input_layer, *hidden_layer_1, *hidden_layer_2,* output_layer, *activation_function);
	}
	cout << " accuracy: " << accuracy(**output_layer, csv_array, examples_number) << endl;
}
void show(struct output_layer output_layer) { //Testing
	for (int i = 0; i < output_layer.nodes_number; i++) {
		cout << output_layer.nodes[i].output_activation;
	}
}
int main() {
	char training_directory[] = "mnist_train.csv";
	int* training_rc_cc_mrl_array = count_rc_cc_mrl(training_directory); //Finds csv file's row count, column count and max row length.
	double** training_array = new double* [training_rc_cc_mrl_array[0] - 1];
	for (int i = 0; i < training_rc_cc_mrl_array[0] - 1; i++) training_array[i] = new double[training_rc_cc_mrl_array[1]]; //Dynamic allocation of training data array (int training_array[60000][785] for given csv file).
	read_csv(training_directory, training_rc_cc_mrl_array[2], training_array); //Reads data of csv file and inserts data into an array.
	char test_directory[] = "mnist_test.csv";
	int* test_rc_cc_mrl_array = count_rc_cc_mrl(training_directory); //Finds csv file's row count, column count and max row length.
	double** test_array = new double* [test_rc_cc_mrl_array[0] - 1];
	for (int i = 0; i < test_rc_cc_mrl_array[0] - 1; i++) test_array[i] = new double[test_rc_cc_mrl_array[1]]; //Dynamic allocation of test data array (int test_array[10000][785] for given csv file).
	read_csv(test_directory, test_rc_cc_mrl_array[2], test_array); //Reads data of csv file and inserts data into an array.
	int layer_count = 4;
	int* next_nodes_number_array = new int[layer_count - 1];
	next_nodes_number_array[0] = 16;
	next_nodes_number_array[1] = 16;
	next_nodes_number_array[2] = 10;
	double learning_rate = 1;
	struct input_layer* input_layer_sigmoid = create_input_layer(training_rc_cc_mrl_array[1] - 1, 16);
	struct hidden_layer* hidden_layer_1_sigmoid = create_hidden_layer(16, 16);
	struct hidden_layer* hidden_layer_2_sigmoid = create_hidden_layer(16, 10);
	struct output_layer* output_layer_sigmoid = create_output_layer(10);
	initialize_ann(training_rc_cc_mrl_array, &input_layer_sigmoid, &hidden_layer_1_sigmoid, &hidden_layer_2_sigmoid, &output_layer_sigmoid, next_nodes_number_array);
	//cout << (*output_layer_sigmoid).nodes[0].output_activation;
	train_ann(&input_layer_sigmoid, &hidden_layer_1_sigmoid, &hidden_layer_2_sigmoid, &output_layer_sigmoid, training_array, training_rc_cc_mrl_array[0] - 1, *sigmoid, *derivative_sigmoid, learning_rate);
	cout << "Sigmoid";
	test_ann(&input_layer_sigmoid, &hidden_layer_1_sigmoid, &hidden_layer_2_sigmoid, &output_layer_sigmoid, test_array, test_rc_cc_mrl_array[0] - 1, *sigmoid);
	delete input_layer_sigmoid;
	delete hidden_layer_1_sigmoid;
	delete hidden_layer_2_sigmoid;
	delete output_layer_sigmoid;
	struct input_layer* input_layer_ReLU = create_input_layer(training_rc_cc_mrl_array[1] - 1, 16);
	struct hidden_layer* hidden_layer_1_ReLU = create_hidden_layer(16, 16);
	struct hidden_layer* hidden_layer_2_ReLU = create_hidden_layer(16, 10);
	struct output_layer* output_layer_ReLU = create_output_layer(10);
	initialize_ann(training_rc_cc_mrl_array, &input_layer_ReLU, &hidden_layer_1_ReLU, &hidden_layer_2_ReLU, &output_layer_ReLU, next_nodes_number_array);
	train_ann(&input_layer_ReLU, &hidden_layer_1_ReLU, &hidden_layer_2_ReLU, &output_layer_ReLU, training_array, training_rc_cc_mrl_array[0] - 1, *ReLU, *derivative_ReLU, learning_rate);
	cout << "ReLU";
	test_ann(&input_layer_ReLU, &hidden_layer_1_ReLU, &hidden_layer_2_ReLU, &output_layer_ReLU, test_array, test_rc_cc_mrl_array[0] - 1, *ReLU);
	delete input_layer_ReLU;
	delete hidden_layer_1_ReLU;
	delete hidden_layer_2_ReLU;
	delete output_layer_ReLU;
	struct input_layer* input_layer_TanH = create_input_layer(training_rc_cc_mrl_array[1] - 1, 16);
	struct hidden_layer* hidden_layer_1_TanH = create_hidden_layer(16, 16);
	struct hidden_layer* hidden_layer_2_TanH = create_hidden_layer(16, 10);
	struct output_layer* output_layer_TanH = create_output_layer(10);
	initialize_ann(training_rc_cc_mrl_array, &input_layer_TanH, &hidden_layer_1_TanH, &hidden_layer_2_TanH, &output_layer_TanH, next_nodes_number_array);
	train_ann(&input_layer_TanH, &hidden_layer_1_TanH, &hidden_layer_2_TanH, &output_layer_TanH, training_array, training_rc_cc_mrl_array[1] - 1, *TanH, *derivative_TanH, learning_rate);
	cout << "TanH";
	test_ann(&input_layer_TanH, &hidden_layer_1_TanH, &hidden_layer_2_TanH, &output_layer_TanH, test_array, test_rc_cc_mrl_array[0] - 1, *TanH);
	delete input_layer_TanH;
	delete hidden_layer_1_TanH;
	delete hidden_layer_2_TanH;
	delete output_layer_TanH;
	for (int i = 0; i < training_rc_cc_mrl_array[0] - 1; i++) delete[]training_array[i];
	delete[]training_array;
	delete[]training_rc_cc_mrl_array;
	delete[]next_nodes_number_array;
	for (int i = 0; i < test_rc_cc_mrl_array[0] - 1; i++) delete[]test_array[i];
	delete[]test_array;
	delete[]test_rc_cc_mrl_array;
	printf("Press any button to exit.");
	char temp=_getch();
	return 0;
}
