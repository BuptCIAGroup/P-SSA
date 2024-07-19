#ifndef GPU_SSA
#define GPU_SSA

#include "stopcondition.h"
#include "common.h"
#include "ssa.h"


struct TSPData {
    const std::vector<std::vector<double>> &dist_matrix_;
    const std::vector<std::vector<double>> &heuristic_matrix_;
    const std::vector<std::vector<uint32_t>> &nn_lists_;
    const uint32_t dimension_;
};

std::map<std::string, pj> 
gpu_run_ssa( TSPData &problem,
             std::mt19937 &rng,
             SSAParams &params,
             StopCondition *stop_cond);

#endif
