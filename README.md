# 損失計算函數

```cpp
__device__ float compute_loss(float predicted, float target) {
    return 0.05f * (predicted - target) * (predicted - target);
}
```
predicted：預測的值（例如，模型計算出來的向量表示）。<br>
target：目標值（例如，對應的真實值或理想值）。<br>
這個函數計算的是平方損失（MSE，Mean Squared Error），即預測值與真實值之間的差異的平方，並將其乘以一個常數因子（0.05f）。<br>
常數 0.05f 用於縮放損失值，這有助於減少權重更新過程中的過度調整，並穩定訓練過程。<br>
這個損失函數用來衡量模型預測結果的誤差，越小的損失值表示模型預測越準確。<br>

# 更新權重的 CUDA 核心函數
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
weights：模型中的嵌入向量（每個單詞對應的向量）。 <br>
context_indices：上下文單詞的索引（每個樣本的上下文詞）。 <br>
target_indices：目標單詞的索引（每個樣本的目標詞）。 <br>
num_samples：總樣本數。 <br>
loss_sum：用於存儲所有樣本的總損失。 <br>

## 流程：

每個 thread 計算一個樣本的損失和權重更新。
target 和 context 是從輸入索引中獲得的單詞索引，並用於查找對應的嵌入向量。
針對每個維度的嵌入向量，根據學習率更新目標和上下文單詞的嵌入向量。
計算每個維度的損失並累加到 sample_loss。
最後，將 sample_loss 累加到總損失 loss_sum 中，並使用 atomicAdd 來保證並行更新不會產生Race Condition。


# 訓練過程函數
```cpp
void train_word2vec_with_loss(int num_epochs, float *d_weights, int *d_context_indices, int *d_target_indices, int num_samples) {
    int blockSize = 512;  // 每個 block 的 thread 數量
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE; // 批次數量
    float *d_loss_sum; // 用於存儲每個批次的損失總和

    checkCudaErrors(cudaMalloc((void**)&d_loss_sum, sizeof(float)));  // 在設備上分配內存

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f; // 每個 epoch 的總損失

        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * BATCH_SIZE;  // 當前批次的起始索引
            int end_idx = min(start_idx + BATCH_SIZE, num_samples);  // 當前批次的結束索引
            int current_batch_size = end_idx - start_idx;  // 當前批次的樣本數量

            // 初始化批次損失為 0
            checkCudaErrors(cudaMemset(d_loss_sum, 0, sizeof(float)));

            // 訓練該批次的數據
            update_weights<<<(current_batch_size + blockSize - 1) / blockSize, blockSize>>>(d_weights, &d_context_indices[start_idx], &d_target_indices[start_idx], current_batch_size, d_loss_sum);
            checkCudaErrors(cudaDeviceSynchronize()); // 同步 CUDA 計算

            // 將批次損失從設備傳回主機
            float batch_loss;
            checkCudaErrors(cudaMemcpy(&batch_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_loss += batch_loss; // 累加批次損失
        }

        // 計算並輸出平均損失
        float avg_loss = epoch_loss / num_samples; // 計算每個 epoch 的平均損失
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << " - Average Loss: " << avg_loss << std::endl;
    }

    cudaFree(d_loss_sum); // 釋放內存
}
```
num_epochs：訓練的總周期數。<br>
d_weights：設備端的權重（嵌入向量）。<br>
d_context_indices 和 d_target_indices：設備端的上下文和目標單詞索引。<br>
num_samples：訓練樣本數量。<br>
blockSize：每個 CUDA block 中的線程數量，用來設置 kernel 的執行規模。<br>
d_loss_sum：設備端的變數，用來存儲每個批次的總損失。<br>

## 流程：

設置批次大小和每個批次的處理數量。
初始化每個 epoch 的總損失為 0。
在每個批次中：
計算並更新該批次的權重。
計算並累加該批次的損失。
每個 epoch 結束後，輸出該 epoch 的平均損失。
