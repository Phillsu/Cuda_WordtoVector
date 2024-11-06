## 損失計算函數

```cpp
__device__ float compute_loss(float predicted, float target) {
    return 0.05f * (predicted - target) * (predicted - target);
}
```
此函數用於計算平方損失（MSE），用來衡量預測值與目標值之間的差異。

