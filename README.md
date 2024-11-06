## 損失計算函數

```cpp
__device__ float compute_loss(float predicted, float target) {
    return 0.05f * (predicted - target) * (predicted - target);
}
\```
