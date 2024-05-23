#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SAMPLE_SIZE 4
#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define LEARNING_RATE 1.0
#define EPOCHS 1000

// example XOR gate that require an hidden layer
float x[SAMPLE_SIZE][INPUT_SIZE] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
float y[SAMPLE_SIZE] = {1, 1, 1, 0};

float random_float();
float sigmoid(float z);
float mse(float y[], float y_pred[]);


int main() {
    int i, j, e, s;

    float y_predicted[SAMPLE_SIZE];

    float w_h[HIDDEN_SIZE][INPUT_SIZE];
    float b_h[HIDDEN_SIZE];
    float z_h[HIDDEN_SIZE];

    float w_o[HIDDEN_SIZE];
    float b_o;
    float z_o;

    // random initialization
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            w_h[i][j] = random_float();
        }
    }

    for (i = 0; i < HIDDEN_SIZE; i++) {
        b_h[i] = random_float();
    }

    for (i = 0; i < HIDDEN_SIZE; i++) {
        w_o[i] = random_float();
    }

    b_o = random_float();

    for (e = 0; e < EPOCHS; e++) {
        /* printf("Epoch %d\n", e+1); */
        for (s = 0; s < SAMPLE_SIZE; s++) {
            // feed forward
            for (i = 0; i < HIDDEN_SIZE; i++) {
                z_h[i] = b_h[i];
                for (j = 0; j < INPUT_SIZE; j++) {
                    z_h[i] += x[s][j] * w_h[i][j];
                } 
                z_h[i] = sigmoid(z_h[i]);
            }

            z_o = b_o;
            for (i = 0; i < HIDDEN_SIZE; i++) {
                z_o += z_h[i] * w_o[i];
            }
            z_o = sigmoid(z_o);
            y_predicted[s] = z_o;
            /* printf("Actual: %f, predicted: %f\n", y[s], y_predicted[s]); */

            // backprop
            // output layer
            // w -= alpha * d_o * z_h 
            float d_o = (z_o - y[s]) * z_o * (1 - z_o);

            for (i = 0; i < HIDDEN_SIZE; i++) {
                w_o[i] -= LEARNING_RATE * d_o * z_h[i];
            }
            b_o -= LEARNING_RATE * d_o;


            // hidden layer
            // w -= alpha * d_h * x
            float d_h[HIDDEN_SIZE];
            for (i = 0; i < HIDDEN_SIZE; i++) {
                d_h[i] = (d_o * w_o[i]) * z_h[i] * (1 - z_h[i]);
            }

            for (i = 0; i < HIDDEN_SIZE; i++) {
                for (j = 0; j < INPUT_SIZE; j++) {
                    w_h[i][j] -= LEARNING_RATE * (d_h[i]) * x[s][j];
                }
            }

            for (i = 0; i < HIDDEN_SIZE; i++) {
                b_h[i] -= LEARNING_RATE * d_h[i];
            }
        }
        printf("Mse: %f\n", mse(y, y_predicted));
    }
    return 0;
}

float random_float() {
	// rand float in range [-1, 1]
    return -1 + 2.0 * ((float)rand() / (float)RAND_MAX);
}

float sigmoid(float s) {
	// activation function
    return 1 / (1 + (1 / exp(s)));
}

// activation function
float mse(float y[], float y_pred[]) {
    float sum = 0.0;
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    return sum / SAMPLE_SIZE;
}
