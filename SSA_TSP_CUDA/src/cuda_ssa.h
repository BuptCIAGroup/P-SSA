#ifndef CUDA_SSA_H
#define CUDA_SSA_H

#include "cuda_utils.h"

struct GPU_SSAParams : public Managed 
{
    float c_ = 0.0;
    float r_ = 1.0;
    float ra_ = 1.0;
    float pchange_ = 0.98;
    
    uint32_t sol_rec_freq_ = 0; 
    uint32_t spiders_count_ = 10;

    GPU_SSAParams &operator=(const SSAParams &other) {
        c_ = other.c_;
        r_ = (float)other.r_;
        ra_ = (float)other.ra_;
        pchange_ = (float)other.pchange_;

        sol_rec_freq_ = static_cast<uint32_t>(other.sol_rec_freq_);
        spiders_count_ = static_cast<uint32_t>(other.spiders_count_);

        return *this;
    }
};

struct Viberation
{
	int src_spider;
	float intensity;
};

struct Spider
{
    uint32_t* best_route;
    uint32_t* curr_route;
    uint32_t* indices_route;
    uint32_t* indices_curr_route;
    
    // float curr_value;
    // float best_value;
    // int inactive_deg;

    int* ex_mask;
   
    Viberation tar;

    float c;
    float r;
    float ra;
    float pchange;
};

#endif
