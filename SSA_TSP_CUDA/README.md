# SSA_TSP_CUDA

This is a new version of the GPU based social spider algorithm for solving the traveling salesman problem (TSP) based on the code of [GPUbasedACS](https://github.com/RSkinderowicz/GPUBasedACS) of Skinderowicz, Rafał.
Thanks for the code support of Skinderowicz, Rafał.


# Compilation Environment 

This code is tested in the following environment:

    Intel i7-7700K 4.20GHz CPU
    RTX2080ti GPU
    Ubuntu 16.04.6 LTS 
    CUDA 10.0
    GCC v5.4



# Building

To compile:

    make

If there are no errors in the compilation, the "gpussa" executable will be generated.


Note that if it doesn't work, you may need to adjust makefil's GPU architecture parameters, for example:

    -gencode arch=compute_50,code=sm_70


# Operating parameters

- --alg: is the name of the algorithm to run.
- --iter: is the number of the SSA iterations to execute,
- --test: is the path to the TSP data instance,
- --outdir: is the path to a directory in which a file with results should be created. Results are saved in JSON format (*.js)

Valid values for the --alg argument:
- ssa_gpu: a fast GPU version of SSA in the way of producer and consumer by using special wap


# Running

If everything goes well, you can do it with the following example:

    ./gpussa --test tsp/usa13509.tsp --outdir results --alg ssa_gpu --iter 100 


You can get most of the parameters with the following command:

    ./gpussa --help
