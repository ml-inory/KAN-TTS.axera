#include "ax_engine.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace kantts {

namespace {

std::vector<char> ReadBinary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void Check(int ret, const char* msg) {
    if (ret != 0) throw std::runtime_error(msg);
}

int g_refcount = 0;

}  // namespace

void AxRuntimeInit() {
    if (g_refcount++ > 0) return;
    Check(AX_SYS_Init(), "AX_SYS_Init failed");
    AX_ENGINE_NPU_ATTR_T attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    Check(AX_ENGINE_Init(&attr), "AX_ENGINE_Init failed");
}

void AxRuntimeDeinit() {
    if (--g_refcount > 0) return;
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
}

struct ModelSession::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_CONTEXT_T context = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io {};
    std::vector<AX_ENGINE_IO_BUFFER_T> inputs;
    std::vector<AX_ENGINE_IO_BUFFER_T> outputs;
    std::vector<char> model;
    std::map<std::string, AX_U32> input_index;
    std::map<std::string, AX_U32> output_index;

    explicit Impl(const std::string& path) : model(ReadBinary(path)) {
        AX_ENGINE_HANDLE_EXTRA_T extra;
        std::memset(&extra, 0, sizeof(extra));
        Check(AX_ENGINE_CreateHandleV2(&handle, model.data(),
                                       static_cast<AX_U32>(model.size()), &extra),
              "AX_ENGINE_CreateHandleV2 failed");
        Check(AX_ENGINE_CreateContextV2(handle, &context), "AX_ENGINE_CreateContextV2 failed");
        Check(AX_ENGINE_GetIOInfo(handle, &info), "AX_ENGINE_GetIOInfo failed");
        inputs.resize(info->nInputSize);
        outputs.resize(info->nOutputSize);
        io.pInputs = inputs.data();
        io.nInputSize = info->nInputSize;
        io.pOutputs = outputs.data();
        io.nOutputSize = info->nOutputSize;
        for (AX_U32 i = 0; i < info->nInputSize; ++i) {
            std::memset(&inputs[i], 0, sizeof(inputs[i]));
            inputs[i].nSize = info->pInputs[i].nSize;
            Check(AX_SYS_MemAllocCached(&inputs[i].phyAddr, &inputs[i].pVirAddr,
                                        inputs[i].nSize, 128, (AX_S8*)"kantts_in"),
                  "input alloc failed");
            input_index[info->pInputs[i].pName] = i;
        }
        for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
            std::memset(&outputs[i], 0, sizeof(outputs[i]));
            outputs[i].nSize = info->pOutputs[i].nSize;
            Check(AX_SYS_MemAllocCached(&outputs[i].phyAddr, &outputs[i].pVirAddr,
                                        outputs[i].nSize, 128, (AX_S8*)"kantts_out"),
                  "output alloc failed");
            output_index[info->pOutputs[i].pName] = i;
        }
    }

    ~Impl() {
        for (auto& x : inputs) if (x.phyAddr) AX_SYS_MemFree(x.phyAddr, x.pVirAddr);
        for (auto& x : outputs) if (x.phyAddr) AX_SYS_MemFree(x.phyAddr, x.pVirAddr);
        if (handle) AX_ENGINE_DestroyHandle(handle);
    }
};

ModelSession::ModelSession(const std::string& model_path) : impl_(new Impl(model_path)) {}
ModelSession::~ModelSession() { delete impl_; }

void ModelSession::SetInput(const std::string& name, const void* data, size_t bytes) {
    auto it = impl_->input_index.find(name);
    if (it == impl_->input_index.end()) throw std::runtime_error("no input named " + name);
    auto& buf = impl_->inputs[it->second];
    if (bytes > buf.nSize) throw std::runtime_error("input too large " + name);
    std::memcpy(buf.pVirAddr, data, bytes);
}

void ModelSession::Run() {
    Check(AX_ENGINE_RunSyncV2(impl_->handle, impl_->context, &impl_->io),
          "AX_ENGINE_RunSyncV2 failed");
}

size_t ModelSession::OutputBytes(const std::string& name) const {
    auto it = impl_->output_index.find(name);
    if (it == impl_->output_index.end()) throw std::runtime_error("no output named " + name);
    return impl_->outputs[it->second].nSize;
}

void ModelSession::GetOutput(const std::string& name, void* out, size_t bytes) const {
    auto it = impl_->output_index.find(name);
    if (it == impl_->output_index.end()) throw std::runtime_error("no output named " + name);
    auto& buf = impl_->outputs[it->second];
    if (bytes > buf.nSize) bytes = buf.nSize;
    std::memcpy(out, buf.pVirAddr, bytes);
}

std::vector<int64_t> ModelSession::OutputShape(const std::string& name) const {
    auto it = impl_->output_index.find(name);
    if (it == impl_->output_index.end()) throw std::runtime_error("no output named " + name);
    auto& t = impl_->info->pOutputs[it->second];
    std::vector<int64_t> shape(t.nShapeSize);
    for (AX_U32 i = 0; i < t.nShapeSize; ++i) shape[i] = t.pShape[i];
    return shape;
}

}  // namespace kantts
