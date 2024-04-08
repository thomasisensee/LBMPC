#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

void DisplayDeviceProperties();
void SetDevice();
int getDevice();
int getDeviceCount();

class CUDATimer {
private:
    cudaEvent_t start, stop;

public:
    CUDATimer();

    ~CUDATimer();

    void startTimer();

    void stopTimer();

    float getElapsedTime();
};

#endif // CUDA_UTILITIES_CUH
