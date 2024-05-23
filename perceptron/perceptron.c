#include <stdio.h>
#include <stdlib.h> // rand
#include <math.h> // exp
#include <time.h> // seed for srand

#define INPUT_SIZE 2
#define SAMPLE_SIZE 4
#define LEARNING_RATE 0.1
#define EPOCHS 1000

float x[SAMPLE_SIZE][INPUT_SIZE] = { // test with AND GATE
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};
float y[SAMPLE_SIZE] = {1, 0, 0, 0};

float random_float(); // generate a random float in [-1, 1] range
float sigmoid(float z); // activation function
float cost(float y, float y_pred);
float mse(float y[], float y_pred[]);

int main() {
    float w[INPUT_SIZE]; // weights
    float b; // bias
    float z; // perceptron output
    float delta; // delta to asjust weights and bias

    // random weights and bias initialization
    srand(time(NULL));
    b = random_float();
    for (int i = 0; i < INPUT_SIZE; i++) {
        w[i] = random_float();
    }
    
    // training of the model
    for (int e = 0; e < EPOCHS; e++) {
		float pred[SAMPLE_SIZE]; // predicted values
        for (int s = 0; s < SAMPLE_SIZE; s++) {
            // feed forward
            // calculate the weighted sum of inputs with weights and sum bias
            z = b;
            for (int i = 0; i < INPUT_SIZE; i++) {
                z += w[i] * x[s][i];
            }

            // pass the weighted sum result trought an activation function
            z = sigmoid(z);
			pred[s] = z;

            // adjust weights and bias
            delta = (z - y[s]);

            b -= LEARNING_RATE * delta;

            for (int i = 0; i < INPUT_SIZE; i++) {
                w[i] -= LEARNING_RATE * delta * x[s][i];
            }
        }
		// print mse
		printf("Cost: %f\n", mse(y, pred));
    }

    return 0;
}

float random_float() {
	// return a random float in [-1, 1] range
    return -1 + 2.0 * ((float)rand() / (float)RAND_MAX);
}

float sigmoid(float z) {
	// activation function
    return 1 / (1 + (exp(-z)));
}

float cost(float y, float y_pred) {
    float d = y - y_pred;
    return 0.5 * d * d;
}

float mse(float y[], float y_pred[]) {
	float sum = 0.0;
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		float d = y[i] - y_pred[i];
		sum += d * d;
	}
	return sum / SAMPLE_SIZE;
}
