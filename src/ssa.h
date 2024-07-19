#ifndef SSA_H
#define SSA_H

#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}


template<typename T, typename R>
static void flatten(const std::vector<std::vector<T>> &matrix, std::vector<R> &out) {
    out.clear();
    for (auto &row : matrix) {
        for (auto el : row) {
            out.push_back((R)el);
        }
    }
}


template<typename T>
std::ostream& operator << (std::ostream &out, const std::vector<T> &vec) {
    for (auto el : vec) {
        out << el << " ";
    }
    return out;
}


extern bool is_valid_route(const std::vector<uint32_t> &route, uint32_t nodes_count);



struct SSAParams 
{
    uint32_t spiders_count_ = 10;
    double c_ = 0.0;
    double r_ = 1.0;
    double ra_ = 1.0;
    double pchange_ = 0.98;
    bool use_local_search_ = false;

    uint32_t sol_rec_freq_ = 0; 

    std::string outdir;
    std::string test_name;
};


#endif
