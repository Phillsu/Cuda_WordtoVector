## 損失計算函數

```cpp
__device__ float compute_loss(float predicted, float target) {
    return 0.05f * (predicted - target) * (predicted - target);
}
```
此函數用於計算平方損失（MSE），用來衡量預測值與目標值之間的差異。

## 更新權重的 CUDA 核心函數
```cpp
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

                // 計算損失
                float predicted = weights[target * EMBEDDING_SIZE + i];
                float actual = 1.0f;
                sample_loss += compute_loss(predicted, actual);
            }
        }
        
        // 使用 atomicAdd 累加損失
        atomicAdd(loss_sum, sample_loss);
    }
}
```
