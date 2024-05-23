#include <stdio.h>
#include <stdlib.h>
#include <math.h> // exp
#include <string.h> // strcmp
#include <time.h> // seed for srand
// example of a simple nn for classification for iris dataset

// SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
#define TRAIN_SIZE 135 // number of sample in the training dataset
#define TEST_SIZE 15 // number of sample in the test dataset
#define INPUT_SIZE 4 // number of features
#define HIDDEN_SIZE 8 // number of hidden neurons
#define CLASSES 3 // number of class of the target
#define LEARNING_RATE 0.1
#define EPOCHS 5000// number of iterations

float random_float();
float sigmoid(float z);
float cost(float y[], float y_pred[]);
void one_hot_encode(float (*vect)[CLASSES], char specie[20]);

int main() {
    FILE *fp;
    int i, j, e, s, id;
    char specie[20];

    /* char classes[CLASSES][20] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}; */

    float x_train[TRAIN_SIZE][INPUT_SIZE]; // train features
    float y_train[TRAIN_SIZE][CLASSES]; // train target

    float x_test[TEST_SIZE][INPUT_SIZE]; // test features
    float y_test[TEST_SIZE][CLASSES]; // test target

    float y_pred[TEST_SIZE][CLASSES]; // predicted values

    float w_h[HIDDEN_SIZE][INPUT_SIZE]; // hidden layer weights
    float b_h[HIDDEN_SIZE]; // hidden layer biases
    float z_h[HIDDEN_SIZE]; // hidden layer outputs

    float w_o[CLASSES][HIDDEN_SIZE]; // output layer weights
    float b_o[CLASSES]; // output layer biases
    float z_o[CLASSES]; // prediction

    // backpropagation deltas
    float d_h[HIDDEN_SIZE];
    float d_o[CLASSES];

    
    // read train data
    fp = fopen("data/train.csv", "r");
    for (i = 0; i < TRAIN_SIZE; i++) {
        fscanf(fp, "%*d,%f,%f,%f,%f,%s\n", &x_train[i][0], &x_train[i][1], &x_train[i][2], &x_train[i][3], specie);
        // create y one hot ecoded
        one_hot_encode(&(y_train[i]), specie);
        
    }
    fclose(fp);

    // read test data
    fp = fopen("data/test.csv", "r");
    for (i = 0; i < TEST_SIZE; i++) {
        fscanf(fp, "%d,%f,%f,%f,%f,%s\n", &id, &x_test[i][0], &x_test[i][1], &x_test[i][2], &x_test[i][3], specie);
        // create y one hot ecoded
        one_hot_encode(&(y_test[i]), specie);
    }
    fclose(fp);


    // random initialization of weights and biases
    srand(time(NULL));
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            w_h[i][j] = random_float();
        }
    }

    for (i = 0; i < HIDDEN_SIZE; i++) {
        b_h[i] = random_float();
    }

    for (i = 0; i < CLASSES; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            w_o[i][j] = random_float();
        }
    }
   
    for (i = 0; i < CLASSES; i++) {
        b_o[i] = random_float();
    }

    // training
    for (e = 0; e < EPOCHS; e++) {
        for (s = 0; s < TRAIN_SIZE; s++) {
            // feed forward 
            // first layer
            for (i = 0; i < HIDDEN_SIZE; i++) {
                z_h[i] = b_h[i];
                for (j = 0; j < INPUT_SIZE; j++) {
                    z_h[i] += w_h[i][j] * x_train[s][j];
                }
                z_h[i] = sigmoid(z_h[i]);
            }

            // output layer
            for (i = 0; i < CLASSES; i++) {
                z_o[i] = b_o[i];
                for (j = 0; j < HIDDEN_SIZE; j++) {
                    z_o[i] += w_o[i][j] * z_h[j];
                }
                z_o[i] = sigmoid(z_o[i]);
            }


            // backprop
            // output layer
            for (i = 0; i < CLASSES; i++) {
                d_o[i] = (z_o[i] - y_train[s][i]) * z_o[i] * (1 - z_o[i]);
            }

            for (i = 0; i < CLASSES; i++) {
                for (j = 0; j < HIDDEN_SIZE; j++) {
                    w_o[i][j] -= LEARNING_RATE * d_o[i] * z_h[j];
                }
            }

            for (i = 0; i < CLASSES; i++) {
                b_o[i] -= LEARNING_RATE * d_o[i];
            }

            // hidden layer
            for (i = 0; i < HIDDEN_SIZE; i++) {
                d_h[i] = 0.0;
                for (j = 0; j < CLASSES; j++) {
                    d_h[i] += d_o[j] * w_o[i][j];
                }
                d_h[i] *= z_h[i] * (1 - z_h[i]);
            }

            for (i = 0; i < HIDDEN_SIZE; i++) {
                for (j = 0; j < INPUT_SIZE; j++) {
                    w_h[i][j] -= LEARNING_RATE * d_h[i] * x_train[s][j];
                }
            }

            for (i = 0; i < HIDDEN_SIZE; i++) {
                b_h[i] -= LEARNING_RATE * d_h[i];
            }
        }
    }
    
    // make prediction about test data
    for (s = 0; s < TEST_SIZE; s++) {
            // feed forward 
            // first layer
            for (i = 0; i < HIDDEN_SIZE; i++) {
                z_h[i] = b_h[i];
                for (j = 0; j < INPUT_SIZE; j++) {
                    z_h[i] += w_h[i][j] * x_test[s][j];
                }
                z_h[i] = sigmoid(z_h[i]);
            }

            // output layer
            for (i = 0; i < CLASSES; i++) {
                z_o[i] = b_o[i];
                for (j = 0; j < HIDDEN_SIZE; j++) {
                    z_o[i] += w_o[i][j] * z_h[j];
                }
                y_pred[s][i] = sigmoid(z_o[i]);
            }

			/* printf("cost: %f\n", cost(y_pred[s], y_test[s])); */
			printf("Actual: [%f %f %f], predicted: [%f %f %f]\n\n", y_test[s][0], y_test[s][1], y_test[s][2], y_pred[s][0], y_pred[s][1], y_pred[s][2]);
    } 

    return 0;
}

float random_float() {
    // generate a random float in the range [-1, 1]
    return -1 + 2.0 * ((float)rand() / (float)RAND_MAX);
}

float sigmoid(float z) {
    // activation function
    return 1 / (1 + (1 / exp(z)));
}

float cost(float y[], float y_pred[]) {
    // calculate the mean squared error of two result
    float sum = 0.0;
    for (int i = 0; i < CLASSES; i++) {
        float d = y[i] - y_pred[i];
        sum += d * d; 
    }
    return sum / CLASSES;
}

void one_hot_encode(float (*vect)[CLASSES], char specie[20]) {
	// one encode species into [1 * 3] vector
	if (strcmp(specie, "Iris-setosa") == 0) {
		(*vect)[0] = 1.0; (*vect)[1] = 0.0; (*vect)[2] = 0.0;
	}
	else if (strcmp(specie, "Iris-versicolor") == 0) {
		(*vect)[0] = 0.0; (*vect)[1] = 1.0; (*vect)[2] = 0.0;
	}
	else{ // iris virginica
		(*vect)[0] = 0.0; (*vect)[1] = 0.0; (*vect)[2] = 1.0;
	}
}
