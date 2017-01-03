/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PServerUtil.h"

namespace paddle {

PServerUtil::PServerUtil(const ParameterServerConfig& config) {
  // round robin to load balance RDMA server ENGINE
  std::vector<std::string> devices;
  int rdmaCpu = 0;
  int onlineCpus = rdma::numCpus();
  int numPorts = config.ports_num() + config.ports_num_for_sparse();

  if (FLAGS_nics.empty()) {
    pservers_.resize(numPorts);
    for (int i = 0; i < numPorts; ++i) {
      if (FLAGS_rdma_tcp == "rdma") {
        pservers_[i].reset(
            new ParameterServer2(std::string(), FLAGS_port + i, rdmaCpu++));
        rdmaCpu = rdmaCpu % onlineCpus;
      } else {
        pservers_[i].reset(new ParameterServer2(std::string(), FLAGS_port + i));
      }
      CHECK(pservers_[i]->init()) << "Fail to initialize parameter server"
                                  << FLAGS_port + i;
    }
  } else {
    str::split(FLAGS_nics, ',', &devices);
    pservers_.resize(devices.size() * numPorts);
    for (int i = 0; i < numPorts; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (FLAGS_rdma_tcp == "rdma") {
          pservers_[i * devices.size() + j].reset(new ParameterServer2(
              getIpAddr(devices[j]), FLAGS_port + i, rdmaCpu++));
          rdmaCpu = rdmaCpu % onlineCpus;
        } else {
          pservers_[i * devices.size() + j].reset(
              new ParameterServer2(getIpAddr(devices[j]), FLAGS_port + i));
        }
        CHECK(pservers_[i * devices.size() + j]->init())
            << "Fail to initialize parameter server" << devices[j]
            << FLAGS_port + i;
      }
    }
  }
}

PServerUtil::~PServerUtil() { this->join(); }

ParameterServerConfig* PServerUtil::initConfig() {
  ParameterServerConfig* config = new ParameterServerConfig();
  config->set_nics(FLAGS_nics);
  config->set_port(FLAGS_port);
  config->set_ports_num(FLAGS_ports_num);
  config->set_rdma_tcp(FLAGS_rdma_tcp);
  return config;
}

PServerUtil* PServerUtil::createWithGflags() {
  auto& pServerConfig = *paddle::PServerUtil::initConfig();
  return create(pServerConfig);
}

PServerUtil* PServerUtil::create(const ParameterServerConfig& config) {
  return new PServerUtil(config);
}

void PServerUtil::start() {
  LOG(INFO) << "pserver sizes : " << pservers_.size();
  int i = 0;
  for (const auto& pserver : pservers_) {
    LOG(INFO) << "pserver started : " << i;
    pserver->start();
    i++;
  }
}

void PServerUtil::join() {
  LOG(INFO) << "pserver sizes : " << pservers_.size();
  int i = 0;
  for (const auto& pserver : pservers_) {
    LOG(INFO) << "pserver join : " << i;
    pserver->join();
    i++;
  }
}

}  // namespace paddle
