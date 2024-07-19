#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <map>
#include <chrono>
#include <cassert>
#include <memory>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "ssa.h"
#include "gpu_ssa.h"
#include "common.h"
#include "stopcondition.h"

#include "cusparse.h"

#include "bupt_global_config.h"

using namespace std;

// Seed with a real random value, if available
std::random_device rd;
std::mt19937 g_rng(rd());


struct TSPProblemData {
    enum EdgeWeightType { NONE, EUC_2D };

    std::vector<std::pair<double, double>> coords_;
    std::map<std::string, std::string> desc_;
    uint32_t dimension_ = 0;
};


static TSPProblemData read_problem_data_tsp(const std::string &path) {
    TSPProblemData pd;
    ifstream file(path);

    if (file.is_open()) {
        string line;
        bool parse_points = false;
        while (getline(file, line)) {
            transform(line.begin(), line.end(), line.begin(), ::tolower);

            if (line.find("eof") != string::npos) {
                parse_points = false;
            } else if (parse_points) {
                istringstream in(line);
                int _;
                double x, y;
                in >> _ >> x >> y;
                pd.coords_.push_back( {x, y} ); 
            } else if (line.find("node_coord_section") != string::npos) {
                parse_points = true;
            } else if (line.find(":") != string::npos) {
                istringstream in(line);                
                string key, value, token;
                in >> key;
                while (in >> token) {
                    if (token != ":") {
                        value += token;
                        value += " ";
                    }
                }
                trim(key);
                trim(value);
                if (key.back() == ':') {
                    key = key.substr(0, key.size() - 1);
                }
                cout << "key: [" << key << "] value: [" << value << "]" << endl;
                pd.desc_[key] = value;
            }
        }

        file.close(); 
    }
    cout << "Coords size: " << pd.coords_.size() << endl;
    pd.dimension_ = pd.coords_.size();
    return pd;
}


static double calc_euc2d_distance(double x1, double y1, double x2, double y2) {
    return (int) ( sqrt( ((x2-x1) * (x2-x1)) + (y2-y1)*(y2-y1) ) + 0.5 );
}


/*
 * Compute ceiling distance between two nodes rounded to next integer for TSPLIB instances.
 */
static double calc_ceil2d_distance(double x1, double y1, double x2, double y2) {
    double xd = x1 - x2;
    double yd = y1 - y2;
    double r = sqrt(xd * xd + yd * yd);
    return (long int)(ceil(r));
}

// static double calc_euc2d_distance(double x1, double y1, double x2, double y2) {
//     return (int) ( ( ((x2-x1) * (x2-x1)) + (y2-y1)*(y2-y1) ) + 0.5 );
// }


// /*
//  * Compute ceiling distance between two nodes rounded to next integer for TSPLIB instances.
//  */
// static double calc_ceil2d_distance(double x1, double y1, double x2, double y2) {
//     double xd = x1 - x2;
//     double yd = y1 - y2;
//     double r = (xd * xd + yd * yd);
//     return (long int)(ceil(r));
// }



template<typename DistanceFunction_t>
static void calc_dist_matrix(DistanceFunction_t calc_distance,
                             const std::vector<std::pair<double, double>> coords,
                             std::vector<std::vector<double>> &matrix
                             ) {
    const auto size = coords.size();
    matrix.resize(size);
    for (auto &row : matrix) {
        row.resize(size, 0.0);
    }
    for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < size; ++j) {
            if (i < j) {
                auto p1 = coords[i];
                auto x1 = p1.first, y1 = p1.second;
                auto p2 = coords[j];
                auto x2 = p2.first, y2 = p2.second;
                auto dist = calc_distance(x1, y1, x2, y2); 
                matrix[i][j] = dist;
                matrix[j][i] = dist; // Symmetric TSP
            }
        }
    }
}


static void calc_dist_matrix(TSPProblemData &problem, std::vector<std::vector<double>> &matrix) {
    if (problem.desc_["edge_weight_type"] == "euc_2d") {
        calc_dist_matrix(calc_euc2d_distance, problem.coords_, matrix);
    } else if (problem.desc_["edge_weight_type"] == "ceil_2d") {
        calc_dist_matrix(calc_ceil2d_distance, problem.coords_, matrix);
    } else {
        throw runtime_error("Unknown edge weight type");
    }
}


static double eval_route(const vector<vector<double>> &dist_matrix,
                  const vector<uint32_t> &route) {
    double r = 0.0;
    if (!route.empty()) {
        uint32_t u = route.back();
        for (uint32_t v : route) {
            r += dist_matrix[u][v];
            u = v;
        }
    }
    return r;
}


template<typename T>
static
std::vector<T> range(T beg, T end) {
    std::vector<T> r;
    for ( ; beg < end; ++beg) { r.push_back(beg); }
    return r;
}

/**
 * Calculate nearest neighbours list for each node
 */
static vector<vector<uint32_t>> 
init_nn_lists(const vector<vector<double>> &dist_matrix, uint32_t nn_size = 32) {
    const uint32_t size = dist_matrix.size();
     
    vector<vector<uint32_t>> nn_lists;
    for (uint32_t i = 0; i < size; ++i) {
        vector<uint32_t> all_nodes = range(0u, size);
        all_nodes.erase( all_nodes.begin() + i ); // We don't want node to be a neighbour of self
        const auto &dist = dist_matrix[i];

        sort(all_nodes.begin(), all_nodes.end(),
            [&](uint32_t u, uint32_t v) -> bool { return dist[u] < dist[v]; });
        
        // assert( dist[all_nodes[0]] <= dist[all_nodes[1]] );
        all_nodes.resize(nn_size); // drop size - nn_size farthest nodes

        nn_lists.push_back(all_nodes);
    }
    return nn_lists;
}


static
vector<uint32_t> build_route_greedy(const vector<vector<double>> &dist_matrix,
        uint32_t start_node = 0) {
    const uint32_t size = dist_matrix.size();

    vector<uint32_t> route;
    route.push_back(start_node);
    auto nodes = range(0u, size);
    nodes.erase( find(nodes.begin(), nodes.end(), start_node) );
    auto curr = route.back();
    while (!nodes.empty()) {
        auto &dist = dist_matrix.at(curr);
        auto closest = *min_element(nodes.begin(), nodes.end(),
                [&] (uint32_t i, uint32_t j) { return dist.at(i) < dist.at(j); });
        route.push_back(closest);
        nodes.erase( find(nodes.begin(), nodes.end(), closest) );
        curr = closest;
    }
    assert(route.size() == size);
    return route;
}

static
void init_heuristic_matrix(const std::vector<std::vector<double>> &dist_matrix,
                           double beta,
                           std::vector<std::vector<double>> &heuristic_matrix) {
    const uint32_t size = dist_matrix.size();
    heuristic_matrix.resize(size);

    for (uint32_t i = 0; i < size; ++i) {
        auto &row = heuristic_matrix[i];
        row.resize(size);
        for (uint32_t j = 0; j < size; ++j) {
            if (i != j) {
                row[j] = 1.0 / pow(dist_matrix[i][j], beta);
            }
        }
    }
}



template<typename T>
T calc_mean(const std::vector<T> &vec) {
    if (vec.empty()) {
        return (T)0;
    }
    T sum = 0;
    for (auto v : vec) {
        sum += v;
    }
    return sum / vec.size();
}


template<typename Fun>
void run_many(Fun f, uint32_t repetitions, std::map< std::string, std::vector<pj> > &all_results) {
    for (uint32_t i = 0; i < repetitions; ++i) {
        cout << "Starting run " << i << endl;
        auto run_start = std::chrono::steady_clock::now();

        auto res = f();

        auto run_end = std::chrono::steady_clock::now();
        auto run_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
        res["run_elapsed"] = pj(run_elapsed.count() / 1.0e6);
        cout << "Run finished in " << (run_elapsed.count() / 1.0e6) << " seconds" << endl;

        for (auto it : res) {
            auto key = it.first;
            auto val = it.second;
            all_results[key].push_back(val);
        }
    }
}


void print_usage() {
    cout << "Program options:\n"
         << setw(20) << left << " --test file "
         << "path to the test file\n";

    cout << setw(20) << left << " --alg name "
         << "Name of the algorithm to run\n";

    cout << setw(20) << left << " --ssa count "
         << "Number of spiders\n";

    cout << setw(20) << left << " --iter count "
         << "Number of iterations per run\n";

    cout << setw(20) << left << " --time_limit seconds "
         << "Time limit per single run in seconds\n";

    cout << setw(20) << left << " --eval_per_node count "
         << "Alternative to --iter & --time_limit. Total num. "
            "of costructed sol. equals dimension * eval_per_node / spiders \n";

    cout << setw(20) << left << " --runs count "
         << "Number of independent algorithm runs\n";

    cout << setw(20) << left << " --pc value "
         << "Value of the pChange SSA parameter\n";

    cout << setw(20) << left << " --outdir path "
         << "Path to the results folder\n";

    cout << setw(20) << left << " --ls [0|1] "
         << "Set to 1 if local search should be used\n";

    cout.flush();
}


string extract_test_name(string path) {
    auto slash_pos = path.rfind("/");
    auto file_name = (slash_pos == string::npos) ? path : path.substr(slash_pos + 1);
    auto dot_pos = file_name.find(".");
    auto test_name = (dot_pos == string::npos) ? file_name : file_name.substr(0, dot_pos);
    return test_name;
}


int main(int argc, char *argv[]) {
    auto init_start = std::chrono::steady_clock::now();

    CmdOptions options(argv, argv + argc);

    if (options.has_option("--help")) {
        print_usage();
        return EXIT_SUCCESS;
    }

    string outdir = options.get_option_or_die("--outdir");

    const string problem_file{ options.get_option_or_die("--test") };
	//获取TSP问题的信息
    auto problem = read_problem_data_tsp(problem_file);
    const uint32_t dimension = problem.coords_.size();

    vector<vector<double>> dist_matrix = { {} };
    calc_dist_matrix(problem, dist_matrix);

    SSAParams params;
	params.outdir = outdir;
	params.test_name = extract_test_name(problem_file);

    params.c_ = options.get_option("--C", 0.0);
    params.r_ = options.get_option("--r", 1.0);
    params.ra_ = options.get_option("--ra", 1.0);
    params.pchange_ = options.get_option("--pc", 0.05);
    params.spiders_count_ = (uint32_t)options.get_option("--ssa", (int)50);

    string alg(options.get_option_or_die("--alg"));
    record("algorithm", alg);

    //solutions record
    params.sol_rec_freq_ = options.get_option("--sol_rec_freq",1);
    record("sol_rec_freq",(int64_t)params.sol_rec_freq_);

    record("start_date_time", get_current_date_time());
    record("problem_file", problem_file);
    record("problem_size", (int64_t)dimension);
    record("c", params.c_);
    record("r", params.r_);
    record("ra", params.ra_);
    record("pchange", params.pchange_);
    record("ssa_count", (int64_t)params.spiders_count_);

    vector<vector<double>> heuristic_matrix;
    init_heuristic_matrix(dist_matrix, 3.0, heuristic_matrix);

    vector<vector<double>> total_matrix;
    vector<vector<uint32_t>> nn_lists;
    { 
        auto start = std::chrono::steady_clock::now();
        nn_lists = init_nn_lists(dist_matrix,NN_SIZE);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "init exact nn time: "<< elapsed.count() / 1.0e6 << std::endl;
        record("init_exact_nn_time", elapsed.count() / 1.0e6);
    }

    auto start = std::chrono::steady_clock::now();

    shared_ptr<StopCondition> stop_cond;
    double time_limit = options.get_option("--time_limit", 0.0);
    record("time_limit", time_limit);
    int iterations = options.get_option("--iter", 0);
    record("iterations", (int64_t)iterations);
    int eval_per_node = options.get_option("--eval_per_node", 0);
    record("eval_per_node", (int64_t)eval_per_node);

    if (time_limit > 0.0) {
        stop_cond.reset(new TimeoutStopCondition(time_limit));
    } else if (iterations > 0) {
        stop_cond.reset(new FixedIterationsStopCondition((uint32_t)iterations));
    } else if (eval_per_node > 0) {
        iterations = (dimension * (size_t)eval_per_node) / params.spiders_count_;
        stop_cond.reset(new FixedIterationsStopCondition((uint32_t)iterations));
    } else {
        cerr << "Number of iterations or time limit is required" << endl;
        return EXIT_FAILURE;
    }

    auto init_end = std::chrono::steady_clock::now();
    auto init_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    //初始化时间
    record("initialization_time", init_elapsed.count() / 1.0e6);
    std::cout << "initialization_time: " << init_elapsed.count() / 1.0e6 << std::endl;

    int ls_option = options.get_option("--ls", 0);
    bool use_local_search = (ls_option == 1);
    params.use_local_search_ = use_local_search;
    record("local_search", (int64_t)ls_option);

    uint32_t runs = (uint32_t)options.get_option("--runs", 1);
    std::map< std::string, std::vector<pj> > all_results;

    if (alg == "ssa_gpu") {
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_ssa( problem, g_rng, params, stop_cond.get());
        };
        run_many(f, runs, all_results);
    } else {
        cerr << "Unknown algorithm [" << alg << "]." << endl;
        return EXIT_FAILURE;
    }

    // Record all algorithm's results
    for (auto it : all_results) {
        auto key = it.first;
        auto val = it.second;
        record(key, val);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Calc. time: " << elapsed.count() / 1.0e6 << " sec" << std::endl;
    record("calc_time", elapsed.count() / 1.0e6);
    // Record command line arguments
    string cmd_args;
    for (int i = 0; i < argc; ++i) {
        cmd_args += argv[i];
        cmd_args += " ";
    }
    record("cmd_args", cmd_args);
    record("end_date_time", get_current_date_time());
    record("experiment", options.get_option("--experiment", "-"));
    record("comment", options.get_option("--comment", "-"));

    auto test_name = extract_test_name(problem_file);
    record("test_name", test_name);

    // Assuming linux OS
    string outfile_path = outdir + "/"
        + "[" + alg + "]" + test_name + "_" 
        + get_current_date_time("%G-%m-%d_%H_%M_%S")
        + "-" + to_string(getpid())
        + ".js";

    cout << outfile_path << endl;

    ofstream outfile(outfile_path);
    if (outfile.is_open()) {
        outfile << global_record_to_string();
        outfile.close();
    }
    return EXIT_SUCCESS;
}
