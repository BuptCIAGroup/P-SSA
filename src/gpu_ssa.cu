#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <utility>
#include <memory>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iomanip>
#include "cuda_utils.h"

#include "common.h"
#include "ssa.h"

#include "gpu_ssa.h"
#include "tsp_ls_gpu.h"
#include "cuda_ssa.h"

#include "bupt_global_config.h"  


__device__ void lock(int *mutex) {
    while( atomicCAS(mutex, 0, 1) != 0 )
        ;
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0); 
}

__device__ int get_global_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void setup_kernel(gpu_rand_state_t *states, uint32_t *seeds) { 
    /* Each thread gets same seed, a different sequence number, no offset */
    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    curand_init(seeds[bid], tid, 0, &states[gid]); 
}


__device__ bool is_node_visited(volatile const uint32_t *visited, int node) {
    return visited[ node >> 5 ] & (1 << (node & 31));
}


__device__ void set_node_visited(volatile uint32_t *visited, int node) {
    visited[ node >> 5 ] |= 1 << (node & 31);
}

__device__ void set_node_un_visited(volatile uint32_t *visited, int node) {
    visited[ node >> 5 ] &= ~(1 << (node & 31));
}


struct DeviceTSPData {
    uint32_t dimension_; 
    DeviceVector<float> heuristic_matrix_;
    DeviceVector<float> dist_matrix_;
    DeviceVector<uint32_t> nn_lists_;

    DeviceTSPData(TSPData &data) 
        : dimension_(data.nn_lists_.size()),
          heuristic_matrix_(data.heuristic_matrix_),
          dist_matrix_(data.dist_matrix_),
          nn_lists_(data.nn_lists_)
    {;}
};

class BaseGPUSSA {
public:

    struct RunContext {
        std::vector<float> spider_values_;
        DeviceVector<float> dev_spider_values_;

        std::vector<float> spider_tmp_values_;
        DeviceVector<float> dev_spider_tmp_values_;

        std::vector<int> masks_;
        DeviceVector<int> dev_masks_;

        std::vector<Spider> spiders_;
        DeviceVector<Spider> dev_spiders_;

        std::vector<uint32_t> curr_routes_;
        DeviceVector<uint32_t> dev_curr_routes_;

        std::vector<uint32_t> indices_routes_;
        DeviceVector<uint32_t> dev_indices_routes_;

        std::vector<uint32_t> indices_curr_routes_;
        DeviceVector<uint32_t> dev_indices_curr_routes_;
        
        std::vector<uint32_t> best_routes_;
        DeviceVector<uint32_t> dev_best_routes_;

        std::vector<float> best_value_;
        DeviceVector<float> dev_best_value_;

        std::vector<uint32_t> best_route_;
        DeviceVector<uint32_t> dev_best_route_;

        std::vector<int> route_node_indices_;
        DeviceVector<int> dev_route_node_indices_;

        DeviceVector<gpu_rand_state_t> dev_prng_states_;

        RunContext(uint32_t dimension, int spiders_count, std::mt19937 &rng) 
            : spider_values_(spiders_count),
              dev_spider_values_(spiders_count),

              spider_tmp_values_(spiders_count),
              dev_spider_tmp_values_(spiders_count),

              spiders_(spiders_count),
              dev_spiders_(spiders_count),

              masks_(spiders_count * spiders_count, 0),
              dev_masks_(masks_),

              curr_routes_(spiders_count * dimension, 0),
              dev_curr_routes_( spiders_count * dimension ),

              indices_routes_(spiders_count * dimension, 0),
              dev_indices_routes_( spiders_count * dimension ),

              indices_curr_routes_(spiders_count * dimension, 0),
              dev_indices_curr_routes_( spiders_count * dimension ),

              best_routes_(spiders_count * dimension, 0),
              dev_best_routes_( spiders_count * dimension ),

              best_value_(1, std::numeric_limits<float>::max()),
              dev_best_value_(best_value_),

              best_route_(dimension),
              dev_best_route_(best_route_),

              route_node_indices_(dimension * spiders_count),
              dev_route_node_indices_(route_node_indices_),

              dev_prng_states_( std::max(WARP_SIZE, spiders_count * WARP_SIZE) )
        {
            init_pseudo_random_number_generator( rng );
        }


        void init_pseudo_random_number_generator( std::mt19937 &rng ) {
            std::vector<uint32_t> seeds( dev_prng_states_.size()/WARP_SIZE);
            std::uniform_int_distribution<> random(0, std::numeric_limits<int>::max());
            for (auto &e : seeds) {
                e = (uint32_t)random( rng );
            }
            DeviceVector<uint32_t> dev_seeds( seeds );
            setup_kernel<<<dev_prng_states_.size()/WARP_SIZE, WARP_SIZE>>>(dev_prng_states_.data(),
                                                         dev_seeds.data());
        }
    };

public:
    BaseGPUSSA(TSPData &problem,
               std::mt19937 &rng,
               SSAParams &params
               )
        : problem_(problem),
          rng_(rng),
          ssa_params_(params) 
    {

    }

    virtual ~BaseGPUSSA() {

    }

    virtual std::map<std::string, pj> run( StopCondition *stop_cond );

    virtual void spider_init();

    virtual void gen_vibration();

    virtual void random_walk();

    virtual void local_search();

    virtual void compare_solution();

    virtual void init_run_context();

    /* Useful for performing tasks just before the run() method finishes */

protected:

    template<typename T>
    void record( const std::string &label, const T &value ) {
        record_[ label ] = pj( value );
    }

    TSPData problem_;
    std::mt19937 &rng_;
    SSAParams ssa_params_;
    std::shared_ptr<DeviceTSPData> dev_problem_ = nullptr;
    std::shared_ptr<RunContext> run_ctx_ = nullptr;
    std::map<std::string, pj> record_; // for reporting / evaluating algorithm
                                       // execution

    std::string iter_solution;
};

void BaseGPUSSA::init_run_context() {

    std::cout << "Initializing context..." << std::endl;

    const int spiders_count = ssa_params_.spiders_count_;
    const uint32_t dimension = problem_.nn_lists_.size();

    run_ctx_.reset( new RunContext(dimension, spiders_count, rng_) );

    std::cout << "...context initialized." << std::endl;
}

/*
   threads_per_block - how many threads per block should be used when building
                       spiders solutions. The number of blocks is equal to the
                       number of spiders.
*/
std::map<std::string, pj>
BaseGPUSSA::run( StopCondition *stop_cond ) {
    using namespace std;

    // Lazy initialization of problem data
    if (dev_problem_ == nullptr) {
        dev_problem_ = make_shared<DeviceTSPData>(problem_);
        std::cout << dev_problem_->dimension_ << std::endl
                  << dev_problem_->heuristic_matrix_.size() << std::endl; 
    }

    std::cout << "Initializing memory..." << std::flush;
    init_run_context();
    std::cout << "memory initialized" << std::endl;

    const int spiders_count = ssa_params_.spiders_count_;
    const uint32_t dimension = problem_.nn_lists_.size();

    vector<uint32_t> temp_route(dimension);

    GPUIntervalTimer sol_timer;
    GPUIntervalTimer ls_timer;
    IntervalTimer move_timer;
    IntervalTimer total_timer;
    total_timer.start_interval();
  
    bool validate_routes = false;

    uint32_t best_found_iteration = 0;

    // Approximate greedy initialization based on NN-List
    spider_init();
    cudaDeviceSynchronize();

    for (stop_cond->init(); !stop_cond->is_reached();
        stop_cond->next_iteration()) {

        move_timer.start_interval();
        sol_timer.start_interval();

        // Generate vibration and update mask
        gen_vibration(); 
        cudaDeviceSynchronize();
        
        // Random walk guided by spiders
        random_walk();
        cudaDeviceSynchronize();

        sol_timer.stop_interval();

        if (ssa_params_.use_local_search_) {
            ls_timer.start_interval();
            // Local search
            local_search();
            ls_timer.stop_interval();
        }
        move_timer.stop_interval();

        // Update best solution and parameters 
        compare_solution();
        cudaDeviceSynchronize();

        // Check if better global solution was found
        run_ctx_->dev_spider_values_.copyTo(run_ctx_->spider_values_);

        if (validate_routes) {
            run_ctx_->dev_best_routes_.copyTo(run_ctx_->best_routes_);
        }

        uint32_t best_index = 0;
        for (uint32_t i = 0; i < spiders_count; ++i) {
            if (run_ctx_->spider_values_[i] < run_ctx_->spider_values_[best_index]) {
                best_index = i;
            }
            if (validate_routes) {
                temp_route.resize(dimension);
                copy(run_ctx_->best_routes_.begin() + i * dimension,
                     run_ctx_->best_routes_.begin() + (i+1) * dimension,
                     temp_route.begin());

                if (!is_valid_route(temp_route, dimension)) {
                    cout << "Invalid route: " << i << endl;
                    cout << temp_route << endl;
                    exit(1);
                }
            }
        }
        
        if (run_ctx_->spider_values_[best_index] < run_ctx_->best_value_[0]) {

            run_ctx_->best_value_[0] = run_ctx_->spider_values_[best_index]; 

            //cout << "New global best found: " << global_best_value << endl;

            run_ctx_->dev_best_routes_.copyTo(run_ctx_->best_routes_);
            copy(run_ctx_->best_routes_.begin() + best_index * dimension,
                 run_ctx_->best_routes_.begin() + (best_index+1) * dimension,
                 run_ctx_->best_route_.begin());

            assert(is_valid_route(run_ctx_->best_route_, dimension));

            run_ctx_->dev_best_route_.copyFrom(run_ctx_->best_route_);
            run_ctx_->dev_best_value_.copyFrom(run_ctx_->best_value_);

            best_found_iteration = stop_cond->get_iteration();
        }

        // Record -- the best solution at present
        if(ssa_params_.sol_rec_freq_ != 0 && stop_cond->get_iteration() % ssa_params_.sol_rec_freq_ == 0){ 
            iter_solution = iter_solution + "[" + to_string(stop_cond->get_iteration()) + \
                                    ": " + to_string((uint32_t)run_ctx_->best_value_[0]) + "] ";
        }

    }


    std::cout << "Final solution: " << run_ctx_->best_value_[0] << endl
              << "Sol. construction time: " << move_timer.get_total_time() << " sec" << std::endl;
    
    total_timer.stop_interval();
    auto total_calc_time = total_timer.get_total_time();
    auto iterations = stop_cond->get_iteration();
    auto iter_time = total_calc_time / iterations;
    auto total_sol_time = sol_timer.get_total_time_ms();
    auto sol_calc_time = total_sol_time / iterations;

    std::cout << "GPU Calc. time: " << total_calc_time << " sec" << std::endl
              << "GPU iteration time [s]: " << iter_time << std::endl
              << "GPU sol. construction time [ms]: " << sol_calc_time << std::endl;

    record( "iterations_made", (int64_t)stop_cond->get_iteration() );
    record( "best_value", run_ctx_->best_value_[0] );
    record( "best_solution", sequence_to_string(run_ctx_->best_route_.begin(),
                                                  run_ctx_->best_route_.end()) );
    record( "best_found_iteration", pj( (int64_t)best_found_iteration ) );
    record( "sol_calc_time_msec", sol_calc_time );
    record( "total_sol_calc_time", total_calc_time );
    record( "iteration_time", iter_time );
    record( "total_local_search_time", ls_timer.get_total_time() );

    if(ssa_params_.sol_rec_freq_ != 0){
        record( "iter_solution", iter_solution);
    }

    return record_;
}


void BaseGPUSSA::local_search() {
    opt2<<<ssa_params_.spiders_count_, WARP_SIZE>>>(
            dev_problem_->dist_matrix_.data(),
            dev_problem_->dimension_,
            run_ctx_->dev_curr_routes_.data(),
            run_ctx_->dev_spider_tmp_values_.data(),
            dev_problem_->dimension_,
            dev_problem_->nn_lists_.data(),
            run_ctx_->dev_route_node_indices_.data()
            );
}

__global__ void
ssa_compare_solution(
    gpu_rand_state_t *rnd_states,
    const GPU_SSAParams ssa_params, 
    uint32_t dimension,
    float *spider_tmp_values, 
    float *spiders_values,
    Spider* spiders)
{
    const uint32_t tid = threadIdx.x;
    // const uint32_t wid = tid / WARP_SIZE;
    const uint32_t lid = tid % WARP_SIZE;
    const uint32_t sid = blockIdx.x * NUM_SPIDER_PER_BLOCK + (tid / WARP_SIZE);
    // const uint32_t num_warps = blockDim.x / WARP_SIZE;
    const bool is_first_in_warp = lid == 0;

    if(spider_tmp_values[sid] < spiders_values[sid]){
        for (uint32_t i = lid; i < dimension; i += WARP_SIZE) {
            spiders[sid].best_route[i] = spiders[sid].curr_route[i];
        }
        __syncwarp();
        if (is_first_in_warp) {
            spiders_values[sid] = spider_tmp_values[sid];
            // spiders[sid].inactive_deg = 0;
            // printf("[%d] %f new \n", sid, spider_tmp_values[sid]);
        }
    }else{
        if (is_first_in_warp) {
            spiders[sid].pchange *= 1.01;
            spiders[sid].ra *= 0.99;
            // spiders[sid].inactive_deg ++;
        }
    }
}

void BaseGPUSSA::compare_solution(){
    GPU_SSAParams gpu_ssa_params;
    gpu_ssa_params = ssa_params_;
    ssa_compare_solution <<<ssa_params_.spiders_count_/NUM_SPIDER_PER_BLOCK, WARP_SIZE * NUM_SPIDER_PER_BLOCK>>>(  
        run_ctx_->dev_prng_states_.data(),
        gpu_ssa_params,
        dev_problem_->dimension_,
        run_ctx_->dev_spider_tmp_values_.data(),
        run_ctx_->dev_spider_values_.data(),
        run_ctx_->dev_spiders_.data()
    );
}

__global__ void
ssa_calc_solution(  gpu_rand_state_t *rnd_states,
                    const GPU_SSAParams ssa_params, 
                    uint32_t dimension,
                    const float * __restrict__ heuristic_matrix,
                    const float * __restrict__ dist_matrix,
                    const uint32_t * __restrict__ nn_lists,
                    uint32_t *best_routes,
                    uint32_t *curr_routes,
                    uint32_t *indices_routes,
                    uint32_t *indices_curr_routes,
                    float *spiders_values,
                    int *masks,
                    Spider* spiders)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t wid = tid / WARP_SIZE;
    const uint32_t lid = tid % WARP_SIZE;
    const uint32_t sid = blockIdx.x * NUM_SPIDER_PER_BLOCK + (tid / WARP_SIZE);
    const uint32_t num_warps = blockDim.x / WARP_SIZE;
    const bool is_first_in_warp = lid == 0;

    assert(num_warps <= NUM_SPIDER_PER_BLOCK);

    PRNG rng(&rnd_states[blockIdx.x*blockDim.x + threadIdx.x]);

    __shared__ volatile uint32_t cand_set[NUM_SPIDER_PER_BLOCK];
    __shared__ volatile float know[NUM_SPIDER_PER_BLOCK];
    __shared__ volatile uint32_t curr_pos[NUM_SPIDER_PER_BLOCK];
    __shared__ volatile uint32_t next_pos[NUM_SPIDER_PER_BLOCK];
    __shared__ volatile uint32_t s_was_visited[MAX_VISITED_SIZE];

    volatile uint32_t* was_visited = s_was_visited + (MAX_VISITED_SIZE / num_warps) * wid;

    Spider* spider = &(spiders[sid]);
    spider->curr_route = curr_routes + sid * dimension;
    spider->best_route = best_routes + sid * dimension;
    spider->indices_route = indices_routes + sid * dimension;
    spider->indices_curr_route = indices_curr_routes + sid * dimension;
    spider->ex_mask = masks + sid * ssa_params.spiders_count_;
    spider->tar.src_spider = sid;
    spider->tar.intensity = 0;
    // spider->inactive_deg = 0;

    spider->c = ssa_params.c_;
    spider->r = ssa_params.r_;
    spider->ra = ssa_params.ra_;
    spider->pchange = ssa_params.pchange_;

    for (int i = tid; i < MAX_VISITED_SIZE; i += blockDim.x) {
        s_was_visited[i] = 0;
    }
    __syncthreads();

    if(is_first_in_warp) {
        curr_pos[wid] = static_cast<uint32_t>(rng.rand_uniform() * dimension) % dimension;
        spider->curr_route[0] = curr_pos[wid];
        spider->best_route[0] = curr_pos[wid];
        spider->indices_route[curr_pos[wid]] = 0;
        spider->indices_curr_route[curr_pos[wid]] = 0;
        set_node_visited(was_visited, curr_pos[wid]);
    }

    for (uint32_t iter = 1; iter < dimension; ++iter) {
        __syncwarp();
        const uint32_t curr = curr_pos[wid];
        next_pos[wid] = dimension;

        const uint32_t *nn = nn_lists + curr * NN_SIZE;
        uint32_t nn_id = nn[lid];
        auto nn_unvisited = !is_node_visited(was_visited, nn_id);

        if (__any(nn_unvisited)) {
            const auto k = dimension * curr + nn_id;
           
            const float nn_know = nn_unvisited 
                ? heuristic_matrix[k] * rng.rand_uniform(): 0;
            
            uint32_t best_nn_id = warp_reduce_arg_max<uint32_t, float, WARP_SIZE>(nn_id, nn_know);

            if(is_first_in_warp) 
                next_pos[wid] = best_nn_id;           
        }

        __syncwarp();

        if (next_pos[wid] == dimension) {
            auto offset = dimension * curr_pos[wid];
            const float *curr_heur = heuristic_matrix + offset;
            float my_max = 0;
            uint32_t my_best_id = dimension;
            for (int i = lid; i < dimension; i += WARP_SIZE) {
                if ( !is_node_visited(was_visited, i) ) {
                        const float product = curr_heur[i] * (rng.rand_uniform() > 0.5 ? 1 : 0.9/*spiders[sid].r*/);
                        my_best_id = (my_max < product) ? i : my_best_id; 
                        my_max = max(my_max, product);
                }
            }
            __syncthreads();
            // Now perform warp reduce
            my_best_id = warp_reduce_arg_max_ext<uint32_t, float, WARP_SIZE>(my_best_id, my_max, &my_max);
            // Now first threads in each warp have indices of best warp elements
            if (is_first_in_warp) {
                next_pos[wid] = my_best_id;
            }
        }
        __syncwarp();

        if (is_first_in_warp) { // Main thread only
            // move ant to the next node
            const uint32_t next_node = next_pos[wid];
            set_node_visited(was_visited, next_node);
            spider->curr_route[iter] = next_node;
            spider->best_route[iter] = next_node;
            spider->indices_route[next_node] = iter;
            spider->indices_curr_route[next_node] = iter;
            curr_pos[wid] = next_node;
        }
    }

    __syncwarp();
    float my_sum = 0;
    for (uint32_t i = lid; i < dimension; i += WARP_SIZE) {
        my_sum += dist_matrix[spider->curr_route[i] * dimension + spider->curr_route[(i + 1) % dimension]];
    }
    __syncwarp();
    // Now perform warp level reduce
    my_sum = warp_reduce<float, WARP_SIZE>(my_sum);
    if (is_first_in_warp) {
        spiders_values[sid] = my_sum;
    }
}

void BaseGPUSSA::spider_init(){
    GPU_SSAParams gpu_ssa_params;
    gpu_ssa_params = ssa_params_;
    ssa_calc_solution <<<ssa_params_.spiders_count_/NUM_SPIDER_PER_BLOCK, WARP_SIZE * NUM_SPIDER_PER_BLOCK>>>(
        run_ctx_->dev_prng_states_.data(),
        gpu_ssa_params,
        dev_problem_->dimension_,
        dev_problem_->heuristic_matrix_.data(),
        dev_problem_->dist_matrix_.data(),
        dev_problem_->nn_lists_.data(),
        run_ctx_->dev_best_routes_.data(),
        run_ctx_->dev_curr_routes_.data(),
        run_ctx_->dev_indices_routes_.data(),
        run_ctx_->dev_indices_curr_routes_.data(),
        run_ctx_->dev_spider_values_.data(),
        run_ctx_->dev_masks_.data(),
        run_ctx_->dev_spiders_.data()
    );
}

__global__ void
ssa_gen_vibration(  gpu_rand_state_t *rnd_states,
                    const GPU_SSAParams ssa_params, 
                    const uint32_t dimension,
                    float *spiders_values,
                    Spider* spiders)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lid = tid % WARP_SIZE;
    const uint32_t sid = blockIdx.x * NUM_SPIDER_PER_BLOCK + (tid / WARP_SIZE);
    const bool is_first_in_warp = lid == 0;

    assert(num_warps <= NUM_SPIDER_PER_BLOCK);

    PRNG rng(&rnd_states[blockIdx.x*blockDim.x + threadIdx.x]);

    float max_intensity = 0.0;
    int max_intensity_id = sid;
    __syncwarp();

    // vibration
    float my_value = spiders_values[sid];
    for (int i=lid; i < ssa_params.spiders_count_; i+=WARP_SIZE)
    {
        float src_value = spiders_values[i];
        if(src_value > my_value || sid == i) 
            continue;

        // float src_intensity = log(1.0 / (float)(src_value - spiders[i].c) + 1.0);

        // float distance = my_value - src_value;
        // float rec_intensity = src_intensity * exp((distance) / spiders[sid].ra) ;

        float rec_intensity = 1.0 / static_cast<float>(src_value - spiders[i].c);

        assert(rec_intensity >= 0.0);

        if (rec_intensity > max_intensity){
            max_intensity = rec_intensity;
            max_intensity_id = i;
        }
    }
    __syncwarp();

    // reduce
    float intensity;
    max_intensity_id = warp_reduce_arg_max_ext<uint32_t, float, WARP_SIZE>(max_intensity_id, max_intensity, &intensity);
    
    if(is_first_in_warp){
        max_intensity = intensity;
        if (max_intensity > spiders[sid].tar.intensity){       
            //printf("GWID %d best id-> %d\n",spiderId,max_intensity_id);
            spiders[sid].tar.src_spider = max_intensity_id;
            spiders[sid].tar.intensity = max_intensity;
            // spiders[sid].inactive_deg = 0;
        }
        // else if(max_intensity_id == sid){
        //     spiders[sid].inactive_deg = -1;
        // }else {
        //     spiders[sid].inactive_deg += 1;
        // }
    }
    __syncwarp();
    
    // update mask
    for (int i=lid; i < ssa_params.spiders_count_; i+=WARP_SIZE){
        spiders[sid].ex_mask[i] = rng.rand_uniform() > spiders[sid].pchange;
    } 
    

    // init (curr, best) route and (curr, best) indices
    for (uint32_t i = lid; i < dimension; i += WARP_SIZE) {
        uint32_t node = spiders[sid].best_route[i];
        spiders[sid].curr_route[i] = node;
        spiders[sid].indices_route[node] = i;
        spiders[sid].indices_curr_route[node] = i;
    }
}

void BaseGPUSSA::gen_vibration(){
    GPU_SSAParams gpu_ssa_params;
    gpu_ssa_params = ssa_params_;
    ssa_gen_vibration <<<ssa_params_.spiders_count_/NUM_SPIDER_PER_BLOCK, WARP_SIZE * NUM_SPIDER_PER_BLOCK>>>(  
        run_ctx_->dev_prng_states_.data(),
        gpu_ssa_params,
        dev_problem_->dimension_,
        run_ctx_->dev_spider_values_.data(),
        run_ctx_->dev_spiders_.data()
    );
}

__global__ void
ssa_random_walk(gpu_rand_state_t *rnd_states,
                const GPU_SSAParams ssa_params, 
                uint32_t dimension,
                const float * __restrict__ heuristic_matrix,
                const float * __restrict__ dist_matrix,
                const uint32_t * __restrict__ nn_lists,
                float *spiders_tmp_values, 
                float *spiders_values,
                Spider* spiders)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t wid = tid / WARP_SIZE;
    const uint32_t lid = tid % WARP_SIZE;
    const uint32_t sid = blockIdx.x * NUM_SPIDER_PER_BLOCK + (tid / WARP_SIZE);
    const uint32_t num_warps = blockDim.x / WARP_SIZE;
    const bool is_first_in_warp = lid == 0;

    assert(num_warps <= NUM_SPIDER_PER_BLOCK);

    PRNG rng(&rnd_states[blockIdx.x*blockDim.x + threadIdx.x]);

    // best spider
    uint32_t src_sid = spiders[sid].tar.src_spider; 

    for(uint32_t iter = 0; iter < dimension; ++iter){

        __syncwarp();

        uint32_t prev_node = spiders[sid].curr_route[(iter+dimension-1)%dimension];
        uint32_t curr_node = spiders[sid].curr_route[iter];
        uint32_t next_node = spiders[sid].curr_route[(iter+1)%dimension];
    
        uint32_t other_sid = static_cast<uint32_t>(rng.rand_uniform() * ssa_params.spiders_count_) % ssa_params.spiders_count_; 
            
        // masked -> using best spider . else random spider  
        uint32_t tmp_curr_node = spiders[sid].ex_mask[other_sid] && src_sid != sid ? \
                        spiders[src_sid].best_route[(spiders[src_sid].indices_route[prev_node]+1)%dimension] : \
                        spiders[other_sid].best_route[(spiders[other_sid].indices_route[prev_node]+1)%dimension];

        float know = 0.0; 
        if(tmp_curr_node != next_node && tmp_curr_node != prev_node && tmp_curr_node != curr_node){
            // exchange prev and next node 
            uint32_t tmp_prev_node = spiders[sid].curr_route[(spiders[sid].indices_curr_route[tmp_curr_node]+dimension-1)%dimension];
            uint32_t tmp_next_node = spiders[sid].curr_route[(spiders[sid].indices_curr_route[tmp_curr_node]+1)%dimension];

            // calc current heuristic
            float my_heur1 = heuristic_matrix[prev_node * dimension + curr_node] + heuristic_matrix[curr_node * dimension + next_node];
            float my_heur2 = heuristic_matrix[tmp_prev_node * dimension + tmp_curr_node] + heuristic_matrix[tmp_curr_node * dimension + tmp_next_node];

            // calc exchange heuristic
            float heur1 = heuristic_matrix[prev_node * dimension + tmp_curr_node] + heuristic_matrix[tmp_curr_node * dimension + next_node];
            float heur2 = heuristic_matrix[tmp_prev_node * dimension + curr_node] + heuristic_matrix[curr_node * dimension + next_node];
            
            // heuristic difference
            know = (heur1+heur2) > (my_heur1+my_heur2) ? (heur1+heur2) - (my_heur1+my_heur2) : 0.0;
        }


        __syncwarp();

        if(__any(know > 0.0)){
            uint32_t best_node = warp_reduce_arg_max<uint32_t, float, WARP_SIZE>(tmp_curr_node, know);

            //exchange
            if(is_first_in_warp) {
                uint32_t index1 = iter;
                uint32_t index2 = spiders[sid].indices_curr_route[best_node];

                spiders[sid].curr_route[index1] = best_node;
                spiders[sid].curr_route[index2] = curr_node;

                spiders[sid].indices_curr_route[curr_node] = index2;
                spiders[sid].indices_curr_route[best_node] = index1;
            }
        }
    }

    float my_sum = 0;
    for (uint32_t i = lid; i < dimension; i += WARP_SIZE) {
        my_sum += dist_matrix[spiders[sid].curr_route[i] * dimension + spiders[sid].curr_route[(i + 1) % dimension]];
    }
    __syncwarp();

    // Now perform warp level reduce
    my_sum = warp_reduce<float, WARP_SIZE>(my_sum);
    if (is_first_in_warp) 
        spiders_tmp_values[sid] = my_sum;
}

void BaseGPUSSA::random_walk(){
    GPU_SSAParams gpu_ssa_params;
    gpu_ssa_params = ssa_params_;
    ssa_random_walk <<<ssa_params_.spiders_count_/NUM_SPIDER_PER_BLOCK, WARP_SIZE * NUM_SPIDER_PER_BLOCK>>>(  
        run_ctx_->dev_prng_states_.data(),
        gpu_ssa_params,
        dev_problem_->dimension_,
        dev_problem_->heuristic_matrix_.data(),
        dev_problem_->dist_matrix_.data(),
        dev_problem_->nn_lists_.data(),
        run_ctx_->dev_spider_tmp_values_.data(),
        run_ctx_->dev_spider_values_.data(),
        run_ctx_->dev_spiders_.data()
    );
}

class GPUSSA : public BaseGPUSSA {
public:

    GPUSSA( TSPData &problem,
            std::mt19937 &rng,
            SSAParams &params )
        : BaseGPUSSA(problem, rng, params)
    {}
    
protected:

};

std::map<std::string, pj> 
gpu_run_ssa( TSPData &problem,
             std::mt19937 &rng,
             SSAParams &params,
             StopCondition *stop_cond ) {

    std::unique_ptr<GPUSSA> alg(new GPUSSA( problem, rng, params ));

    return alg->run(stop_cond);
}



