#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include <omp.h>


namespace std{
    namespace
    {

        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>>
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {
            size_t seed = 0;
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
            return seed;
        }

    };
}

int add(int n, std::vector<int>& arr) {
    int res = 0;
    for (int i = 0; i < n; ++i) {
        res += arr[i];
    }
    return res; //n + n1;
}

/* Random walk from each node. */
std::unordered_map<std::tuple<int,int>, int> random_walks(const std::map<int, std::vector<int>>& neighbors,
                                            const int walk_length, const int num_trials){
  // contains random walk for each node across trials
  std::unordered_map<std::tuple<int,int>, int> output;

    #pragma omp parallel
    {
        size_t cnt = 0;
        for(auto node = neighbors.begin(); node !=neighbors.end(); ++node, cnt++)
        { 
            // do walk for each of this curr_node
            int curr_node = node->first;
            for (int i=0; i < num_trials; i++){
                std::vector<float> walk;
                walk.push_back(curr_node);
                int last_visited_node = walk.back();
                std::vector<int> curr_neigh = neighbors.at(last_visited_node);
                // select random element from the neighbors
                int nex = curr_neigh[rand() % curr_neigh.size()];
                walk.push_back(nex);
                // update count in the global return map
                std::tuple<int, int> dict_key = std::make_tuple(curr_node, nex);
                output[dict_key]++ ;

                while(walk.size() < walk_length){
                    int walk_size = walk.size();
                    int cur = walk[walk_size - 1];
                    int prev = walk[walk_size - 2];
                    std::vector<int> curr_neigh = neighbors.at(cur);
                    std::vector<int> copy_curr_neigh;
                    copy_curr_neigh.assign(curr_neigh.begin(), curr_neigh.end());
                    copy_curr_neigh.erase(std::remove(copy_curr_neigh.begin(), copy_curr_neigh.end(), prev),
                                            copy_curr_neigh.end());
                    if (copy_curr_neigh.size() < 1) {
                        break;
                    }
                    // select random element from the neighbors
                    int nex = copy_curr_neigh[rand() % copy_curr_neigh.size()];
                    // update count in the global return map. curr_node is the walk start node here
                    std::tuple<int, int> dict_key = std::make_tuple(curr_node, nex);
                    output[dict_key]++ ;

                    // append nex to the walk
                    walk.push_back(nex);
                }

            }
        }
  }
  return output;
}

namespace py = pybind11;

PYBIND11_MODULE(rwcpp, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("random_walks", &random_walks, "Trial function");
}
