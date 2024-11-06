#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>  // For square loss computation
#include <cuda_runtime.h>
#include <unordered_map>
#include <sstream>
#include <random>

const int EMBEDDING_SIZE = 300;    // Dimension of vectors
const float LEARNING_RATE = 0.01f; // Learning rate
const int WINDOW_SIZE = 10;         // Context window size
const int VOCAB_SIZE = 1000000;     // Vocabulary size
const int BATCH_SIZE = 1024;       // Batch size

// CUDA error checking
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Compute Mean Squared Error Loss
__device__ float compute_loss(float predicted, float target) {
    return 0.05f * (predicted - target) * (predicted - target);
}

// Update weights and compute loss
__global__ void update_weights(float *weights, const int *context_indices, const int *target_indices, int num_samples, float *loss_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int target = target_indices[idx];
        int context = context_indices[idx];
        
        float sample_loss = 0.0f;

        if (target >= 0 && target < VOCAB_SIZE && context >= 0 && context < VOCAB_SIZE) {
            for (int i = 0; i < EMBEDDING_SIZE; i++) {
                float grad = LEARNING_RATE * (1.0f - weights[target * EMBEDDING_SIZE + i]);
                weights[target * EMBEDDING_SIZE + i] += grad;
                weights[context * EMBEDDING_SIZE + i] += grad;

                // Compute loss
                float predicted = weights[target * EMBEDDING_SIZE + i];
                float actual = 1.0f;
                sample_loss += compute_loss(predicted, actual);
            }
        }
        
        // Accumulate loss into global variable
        atomicAdd(loss_sum, sample_loss);
    }
}

// Train the model over multiple epochs with batch training
void train_word2vec_with_loss(int num_epochs, float *d_weights, int *d_context_indices, int *d_target_indices, int num_samples) {
    int blockSize = 512;
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    float *d_loss_sum;

    checkCudaErrors(cudaMalloc((void**)&d_loss_sum, sizeof(float)));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * BATCH_SIZE;
            int end_idx = min(start_idx + BATCH_SIZE, num_samples);
            int current_batch_size = end_idx - start_idx;

            // Initialize batch loss_sum
            checkCudaErrors(cudaMemset(d_loss_sum, 0, sizeof(float)));

            // Train batch data
            update_weights<<<(current_batch_size + blockSize - 1) / blockSize, blockSize>>>(d_weights, &d_context_indices[start_idx], &d_target_indices[start_idx], current_batch_size, d_loss_sum);
            checkCudaErrors(cudaDeviceSynchronize());

            // Accumulate batch loss into epoch_loss
            float batch_loss;
            checkCudaErrors(cudaMemcpy(&batch_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_loss += batch_loss;
        }

        // Compute average epoch loss
        float avg_loss = epoch_loss / num_samples;
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << " - Average Loss: " << avg_loss << std::endl;
    }

    cudaFree(d_loss_sum);
}

// Helper function to read sentences from a file
std::vector<std::string> read_sentences_from_file(const std::string& filename) {
    std::vector<std::string> sentences;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        sentences.push_back(line);
    }
    return sentences;
}

// Build vocabulary and count frequencies
std::unordered_map<std::string, int> build_vocab_and_count_frequencies(const std::vector<std::string>& sentences) {
    std::unordered_map<std::string, int> vocab;
    int index = 0;

    for (const auto& sentence : sentences) {
        std::istringstream iss(sentence);
        std::string word;
        while (iss >> word) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = index++;
            }
        }
    }

    return vocab;
}

// Function to print each word's vector to file
void print_device_weights_to_file(float *d_weights, int vocab_size, int embedding_size, const std::string& filename) {
    // Allocate host memory for the weights
    std::vector<float> h_weights(vocab_size * embedding_size);

    // Copy the weights from device to host
    checkCudaErrors(cudaMemcpy(h_weights.data(), d_weights, vocab_size * embedding_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Open the file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write each word's vector to the file
    for (int i = 0; i < vocab_size; i++) {
        file << "Word " << i << " weights: [";
        for (int j = 0; j < embedding_size; j++) {
            file << h_weights[i * embedding_size + j];
            if (j < embedding_size - 1) file << ", ";
        }
        file << "]\n";
    }

    // Close the file
    file.close();
    std::cout << "Vectors saved to " << filename << std::endl;
}

int main() {
    // Read text data
    std::vector<std::string> sentences = read_sentences_from_file("dataset/HarryPotter1.txt");

    // Build vocabulary
    std::unordered_map<std::string, int> vocab = build_vocab_and_count_frequencies(sentences);
    
    // Calculate number of samples
    int num_samples = 0;
    std::vector<int> context_indices; // Declare context_indices
    std::vector<int> target_indices;  // Declare target_indices

    for (const auto& sentence : sentences) {
        std::istringstream iss(sentence);
        std::vector<int> indices;
        std::string word;
        while (iss >> word) {
            if (vocab.find(word) != vocab.end()) {
                indices.push_back(vocab[word]);
            }
        }

        for (size_t i = 0; i < indices.size(); i++) {
            for (int j = 1; j <= WINDOW_SIZE; j++) {
                if (i >= j) {
                    context_indices.push_back(indices[i - j]);
                    target_indices.push_back(indices[i]);
                    num_samples++;
                }
                if (i + j < indices.size()) {
                    context_indices.push_back(indices[i + j]);
                    target_indices.push_back(indices[i]);
                    num_samples++;
                }
            }
        }
    }

    // Initialize device memory
    float *d_weights;
    int *d_context_indices, *d_target_indices;
    checkCudaErrors(cudaMalloc((void**)&d_weights, VOCAB_SIZE * EMBEDDING_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_context_indices, num_samples * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_target_indices, num_samples * sizeof(int)));

    // Initialize weights
    std::vector<float> weights(VOCAB_SIZE * EMBEDDING_SIZE);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
    for (auto& w : weights) {
        w = distribution(generator);
    }
    checkCudaErrors(cudaMemcpy(d_weights, weights.data(), VOCAB_SIZE * EMBEDDING_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Copy context and target indices to device
    checkCudaErrors(cudaMemcpy(d_context_indices, context_indices.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_target_indices, target_indices.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice));

    // Train the model
    int num_epochs = 100; // Set number of epochs
    train_word2vec_with_loss(num_epochs, d_weights, d_context_indices, d_target_indices, num_samples);

    // Print each word's vector to vector.txt
    print_device_weights_to_file(d_weights, VOCAB_SIZE, EMBEDDING_SIZE, "vector.txt");

    // Free device memory
    cudaFree(d_weights);
    cudaFree(d_context_indices);
    cudaFree(d_target_indices);

    return 0;
}
