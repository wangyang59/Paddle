/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "hl_dso_loader.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/CommandLineParser.h"

P_DEFINE_string(cudnn_dir, "",
                "Specify path for loading libcudnn.so. For instance, "
                "/usr/local/cudnn/lib64. If empty [default], dlopen will search "
                "cudnn from LD_LIBRARY_PATH");

P_DEFINE_string(cuda_dir, "",
                "Specify path for loading cuda library, such as libcublas, "
                "libcurand. For instance, /usr/local/cuda/lib64. "
                "(Note: libcudart can not be specified by cuda_dir, since some "
                "build-in function in cudart already ran before main entry). "
                "If empty [default], dlopen will search cuda from LD_LIBRARY_PATH");

static inline std::string join(const std::string& part1, const std::string& part2) {
  // directory separator
  const char sep = '/';

  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline void GetDsoHandleFromDefaultPath(
        std::string& dso_path, void** dso_handle, int dynload_flags) {
    LOG(INFO) << "Try to find cuda library: " << dso_path
              << "from default system path.";
    // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH 
    *dso_handle = dlopen(dso_path.c_str(), dynload_flags);
    
    // DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
    // bring System Integrity Projection (SIP), if dso_handle
    // is null, search from default package path in Mac OS.
    #if defined(__APPLE__) or defined(__OSX__)
    if (nullptr == *dso_handle) {
        dso_path = join("/usr/local/cuda/lib/", dso_path);
        *dso_handle = dlopen(dso_path.c_str(), dynload_flags);
        if (nullptr == *dso_handle) {
            if (dso_path == "libcudnn.dylib") {
                LOG(FATAL) << "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n"
                << "For instance, sudo tar -xzf cudnn-7.5-osx-x64-v5.0-ga.tgz -C "
                << "/usr/local \n sudo chmod a+r /usr/local/cuda/include/cudnn.h "
                << "/usr/local/cuda/lib/libcudnn*";
            }
        } 
    }   
    #endif
}

static inline void GetDsoHandleFromSearchPath(
        const std::string& search_root,
        const std::string& dso_name,
        void** dso_handle) {
    int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
    *dso_handle = nullptr;

    std::string dlPath = dso_name;
    if (search_root.empty()) {
        GetDsoHandleFromDefaultPath(dlPath, dso_handle, dynload_flags);
    } else {
        // search xxx.so from custom path
        dlPath = join(search_root, dso_name);
        *dso_handle = dlopen(dlPath.c_str(), dynload_flags);
        // if not found, search from default path
        if (nullptr == dso_handle) {
            LOG(WARNING) << "Failed to find cuda library: " << dlPath;
            dlPath = dso_name;
            GetDsoHandleFromDefaultPath(dlPath, dso_handle, dynload_flags);
        }
    }

    CHECK(nullptr != *dso_handle)
      << "Failed to find cuda library: " << dlPath << std::endl
      << "Please specify its path correctly using one of the following ideas: \n"

      << "Idea 1. set cuda and cudnn lib path at runtime. "
      << "http://www.paddlepaddle.org/doc/ui/cmd_argument/argument_outline.html \n"
      << "For instance, issue command: paddle train --use_gpu=1 "
      << "--cuda_dir=/usr/local/cudnn/lib --cudnn_dir=/usr/local/cudnn/lib ...\n"

      << "Idea 2. set environment variable LD_LIBRARY_PATH on Linux or "
      << "DYLD_LIBRARY_PATH on Mac OS. \n"
      << "For instance, issue command: export LD_LIBRARY_PATH=... \n"

      << "Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is impossible "
      << "unless System Integrity Protection (SIP) is disabled. However, @Idea 1"
      << "always work well.";
}

void GetCublasDsoHandle(void** dso_handle) {
#if defined(__APPLE__) || defined(__OSX__)
    GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.dylib", dso_handle);
#else
    GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.so", dso_handle);
#endif
}

void GetCudnnDsoHandle(void** dso_handle) {
#if defined(__APPLE__) || defined(__OSX__)
    GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.dylib", dso_handle);
#else
    GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.so", dso_handle);
#endif
}

void GetCudartDsoHandle(void** dso_handle) {
#if defined(__APPLE__) || defined(__OSX__)
    GetDsoHandleFromSearchPath("", "libcudart.dylib", dso_handle);
#else
    GetDsoHandleFromSearchPath("", "libcudart.so", dso_handle);
#endif
}

void GetCurandDsoHandle(void** dso_handle) {
#if defined(__APPLE__) || defined(__OSX__)
    GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.dylib", dso_handle);
#else
    GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.so", dso_handle);
#endif
}
