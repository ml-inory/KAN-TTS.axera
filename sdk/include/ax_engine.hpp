#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace kantts {

// AXMODEL 推理会话（AXEngine C API，多输入/多输出，按张量名存取字节缓冲区）。
class ModelSession {
public:
    explicit ModelSession(const std::string& model_path);
    ~ModelSession();
    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    void SetInput(const std::string& name, const void* data, size_t bytes);
    void Run();
    size_t OutputBytes(const std::string& name) const;
    void GetOutput(const std::string& name, void* out, size_t bytes) const;
    std::vector<int64_t> OutputShape(const std::string& name) const;

private:
    struct Impl;
    Impl* impl_;
};

// 全局 AX 运行时初始化（进程内一次）。
void AxRuntimeInit();
void AxRuntimeDeinit();

}  // namespace kantts
