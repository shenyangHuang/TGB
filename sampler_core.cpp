#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef float TimeStampType;

class TemporalGraphBlock
{
public:
    std::vector<NodeIDType> row;
    std::vector<NodeIDType> col;
    std::vector<EdgeIDType> eid;
    std::vector<TimeStampType> ts;
    std::vector<TimeStampType> dts;
    std::vector<NodeIDType> nodes;
    NodeIDType dim_in, dim_out;
    double ptr_time = 0;
    double search_time = 0;
    double sample_time = 0;
    double tot_time = 0;
    double coo_time = 0;

    TemporalGraphBlock() {}
    TemporalGraphBlock(std::vector<NodeIDType> &_row, std::vector<NodeIDType> &_col,
                       std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                       std::vector<TimeStampType> &_dts, std::vector<NodeIDType> &_nodes,
                       NodeIDType _dim_in, NodeIDType _dim_out) : row(_row), col(_col), eid(_eid), ts(_ts), dts(_dts),
                                                                  nodes(_nodes), dim_in(_dim_in), dim_out(_dim_out) {}
};

class ParallelSampler
{
public:
    std::vector<EdgeIDType> indptr;
    std::vector<EdgeIDType> indices;
    std::vector<EdgeIDType> eid;
    std::vector<TimeStampType> ts;
    NodeIDType num_nodes;
    EdgeIDType num_edges;
    int num_thread_per_worker;
    int num_workers;
    int num_threads;
    int num_layers;
    std::vector<int> num_neighbors;
    bool recent;
    bool prop_time;
    int num_history;
    TimeStampType window_duration;
    std::vector<std::vector<std::vector<EdgeIDType>::size_type>> ts_ptr;
    omp_lock_t *ts_ptr_lock;
    std::vector<TemporalGraphBlock> ret;

    ParallelSampler(std::vector<EdgeIDType> &_indptr, std::vector<EdgeIDType> &_indices,
                    std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                    int _num_thread_per_worker, int _num_workers, int _num_layers,
                    std::vector<int> &_num_neighbors, bool _recent, bool _prop_time,
                    int _num_history, TimeStampType _window_duration) : indptr(_indptr), indices(_indices), eid(_eid), ts(_ts), prop_time(_prop_time),
                                                                        num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
                                                                        num_layers(_num_layers), num_neighbors(_num_neighbors), recent(_recent),
                                                                        num_history(_num_history), window_duration(_window_duration)
    {
        omp_set_num_threads(num_thread_per_worker * num_workers);
        num_threads = num_thread_per_worker * num_workers;
        num_nodes = indptr.size() - 1;
        num_edges = indices.size();
        ts_ptr_lock = (omp_lock_t *)malloc(num_nodes * sizeof(omp_lock_t));
        for (int i = 0; i < num_nodes; i++)
            omp_init_lock(&ts_ptr_lock[i]);
        ts_ptr.resize(num_history + 1);
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1);
#pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }

    void reset()
    {
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1);
#pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }

    void update_ts_ptr(int slc, std::vector<NodeIDType> &root_nodes,
                       std::vector<TimeStampType> &root_ts, float offset)
    {
#pragma omp parallel for schedule(static, int(ceil(static_cast <float>(root_nodes.size()) / num_threads)))
        for (std::vector<NodeIDType>::size_type i = 0; i < root_nodes.size(); i++)
        {
            NodeIDType n = root_nodes[i];
            omp_set_lock(&(ts_ptr_lock[n]));
            for (std::vector<EdgeIDType>::size_type j = ts_ptr[slc][n]; j < indptr[n + 1]; j++)
            {
                // std::cout << "comparing " << ts[j] << " with " << root_ts[i] << std::endl;
                if (ts[j] > (root_ts[i] + offset - 1e-7f))
                {
                    if (j != ts_ptr[slc][n])
                        ts_ptr[slc][n] = j - 1;
                    break;
                }
                if (j == indptr[n + 1] - 1)
                {
                    ts_ptr[slc][n] = j;
                }
            }
            omp_unset_lock(&(ts_ptr_lock[n]));
        }
    }

    inline void add_neighbor(std::vector<NodeIDType> *_row, std::vector<NodeIDType> *_col,
                             std::vector<EdgeIDType> *_eid, std::vector<TimeStampType> *_ts,
                             std::vector<TimeStampType> *_dts, std::vector<NodeIDType> *_nodes,
                             EdgeIDType &k, TimeStampType &src_ts, int &row_id)
    {
        _row->push_back(row_id);
        _col->push_back(_nodes->size());
        _eid->push_back(eid[k]);
        if (prop_time)
            _ts->push_back(src_ts);
        else
            _ts->push_back(ts[k]);
        _dts->push_back(src_ts - ts[k]);
        _nodes->push_back(indices[k]);
        // _row.push_back(0);
        // _col.push_back(0);
        // _eid.push_back(0);
        // if (prop_time)
        //     _ts.push_back(src_ts);
        // else
        //     _ts.push_back(10000);
        // _nodes.push_back(100);
    }

    inline void combine_coo(TemporalGraphBlock &_ret, std::vector<NodeIDType> **_row,
                            std::vector<NodeIDType> **_col,
                            std::vector<EdgeIDType> **_eid,
                            std::vector<TimeStampType> **_ts,
                            std::vector<TimeStampType> **_dts,
                            std::vector<NodeIDType> **_nodes,
                            std::vector<int> &_out_nodes)
    {
        std::vector<EdgeIDType> cum_row, cum_col;
        cum_row.push_back(0);
        cum_col.push_back(0);
        for (int tid = 0; tid < num_threads; tid++)
        {
            // std::cout<<tid<<" here "<<_out_nodes[tid]<<std::endl;
            cum_row.push_back(cum_row.back() + _out_nodes[tid]);
            cum_col.push_back(cum_col.back() + _col[tid]->size());
        }
        int num_root_nodes = _ret.nodes.size();
        _ret.row.resize(cum_col.back());
        _ret.col.resize(cum_col.back());
        _ret.eid.resize(cum_col.back());
        _ret.ts.resize(cum_col.back() + num_root_nodes);
        _ret.dts.resize(cum_col.back() + num_root_nodes);
        _ret.nodes.resize(cum_col.back() + num_root_nodes);
#pragma omp parallel for schedule(static, 1)
        for (int tid = 0; tid < num_threads; tid++)
        {
            std::transform(_row[tid]->begin(), _row[tid]->end(), _row[tid]->begin(),
                           [&](auto &v)
                           { return v + cum_row[tid]; });
            std::transform(_col[tid]->begin(), _col[tid]->end(), _col[tid]->begin(),
                           [&](auto &v)
                           { return v + cum_col[tid] + num_root_nodes; });
            std::copy(_row[tid]->begin(), _row[tid]->end(), _ret.row.begin() + cum_col[tid]);
            std::copy(_col[tid]->begin(), _col[tid]->end(), _ret.col.begin() + cum_col[tid]);
            std::copy(_eid[tid]->begin(), _eid[tid]->end(), _ret.eid.begin() + cum_col[tid]);
            std::copy(_ts[tid]->begin(), _ts[tid]->end(), _ret.ts.begin() + cum_col[tid] + num_root_nodes);
            std::copy(_dts[tid]->begin(), _dts[tid]->end(), _ret.dts.begin() + cum_col[tid] + num_root_nodes);
            std::copy(_nodes[tid]->begin(), _nodes[tid]->end(), _ret.nodes.begin() + cum_col[tid] + num_root_nodes);
            delete _row[tid];
            delete _col[tid];
            delete _eid[tid];
            delete _ts[tid];
            delete _dts[tid];
            delete _nodes[tid];
        }
        _ret.dim_in = _ret.nodes.size();
        _ret.dim_out = cum_row.back();
    }

    void sample_layer(std::vector<NodeIDType> &_root_nodes, std::vector<TimeStampType> &_root_ts,
                      int neighs, bool use_ptr, bool from_root)
    {
        double t_s = omp_get_wtime();
        std::vector<NodeIDType> *root_nodes;
        std::vector<TimeStampType> *root_ts;
        if (from_root)
        {
            root_nodes = &_root_nodes;
            root_ts = &_root_ts;
        }
        double t_ptr_s = omp_get_wtime();
        if (use_ptr)
            update_ts_ptr(num_history, *root_nodes, *root_ts, 0);
        ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
        for (int i = 0; i < num_history; i++)
        {
            if (!from_root)
            {
                root_nodes = &(ret[ret.size() - 1 - i - num_history].nodes);
                root_ts = &(ret[ret.size() - 1 - i - num_history].ts);
            }
            TimeStampType offset = -i * window_duration;
            t_ptr_s = omp_get_wtime();
            if ((use_ptr) && (std::abs(window_duration) > 1e-7f))
                update_ts_ptr(num_history - 1 - i, *root_nodes, *root_ts, offset - window_duration);
            ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
            std::vector<NodeIDType> *_row[num_threads];
            std::vector<NodeIDType> *_col[num_threads];
            std::vector<EdgeIDType> *_eid[num_threads];
            std::vector<TimeStampType> *_ts[num_threads];
            std::vector<TimeStampType> *_dts[num_threads];
            std::vector<NodeIDType> *_nodes[num_threads];
            std::vector<int> _out_node(num_threads, 0);
            int reserve_capacity = int(ceil((*root_nodes).size() / num_threads)) * neighs;
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                unsigned int loc_seed = tid;
                _row[tid] = new std::vector<NodeIDType>;
                _col[tid] = new std::vector<NodeIDType>;
                _eid[tid] = new std::vector<EdgeIDType>;
                _ts[tid] = new std::vector<TimeStampType>;
                _dts[tid] = new std::vector<TimeStampType>;
                _nodes[tid] = new std::vector<NodeIDType>;
                _row[tid]->reserve(reserve_capacity);
                _col[tid]->reserve(reserve_capacity);
                _eid[tid]->reserve(reserve_capacity);
                _ts[tid]->reserve(reserve_capacity);
                _dts[tid]->reserve(reserve_capacity);
                _nodes[tid]->reserve(reserve_capacity);
// #pragma omp critical
//                     std::cout<<tid<<" sampling: "<<root_nodes->size()<<" "<<int(ceil((*root_nodes).size() / num_threads))<<std::endl;
#pragma omp for schedule(static, int(ceil(static_cast <float>((*root_nodes).size()) / num_threads)))
                for (std::vector<NodeIDType>::size_type j = 0; j < (*root_nodes).size(); j++)
                {
                    NodeIDType n = (*root_nodes)[j];
                    // if (tid == 16)
                    //     std::cout << _out_node[tid] << " " <<j << " " << n << std::endl;
                    TimeStampType nts = (*root_ts)[j];
                    EdgeIDType s_search, e_search;
                    if (use_ptr)
                    {
                        s_search = ts_ptr[num_history - 1 - i][n];
                        e_search = ts_ptr[num_history - i][n];
                    }
                    else
                    {
                        // search for start and end pointer
                        double t_search_s = omp_get_wtime();
                        if (num_history == 1)
                        {
                            // TGAT style
                            s_search = indptr[n];
                            auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                                         ts.begin() + indptr[n + 1], nts);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        else
                        {
                            // DySAT style
                            auto s_it = std::upper_bound(ts.begin() + indptr[n],
                                                         ts.begin() + indptr[n + 1],
                                                         nts + offset - window_duration);
                            s_search = std::max(int(s_it - ts.begin()) - 1, indptr[n]);
                            auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                                         ts.begin() + indptr[n + 1], nts + offset);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        if (tid == 0)
                            ret[0].search_time += omp_get_wtime() - t_search_s;
                    }
                    // std::cout << n << " " << s_search << " " << e_search << std::endl;
                    double t_sample_s = omp_get_wtime();
                    if ((recent) || (e_search - s_search < neighs))
                    {
                        // no sampling, pick recent neighbors
                        for (EdgeIDType k = e_search; k > std::max(s_search, e_search - neighs); k--)
                        {
                            if (ts[k] < nts + offset - 1e-7f)
                            {
                                add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                             _dts[tid], _nodes[tid], k, nts, _out_node[tid]);
                            }
                        }
                    }
                    else
                    {
                        // random sampling within ptr
                        for (int _i = 0; _i < neighs; _i++)
                        {
                            EdgeIDType picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
                            if (ts[picked] < nts + offset - 1e-7f)
                            {
                                add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                             _dts[tid], _nodes[tid], picked, nts, _out_node[tid]);
                            }
                        }
                    }
                    _out_node[tid] += 1;
                    if (tid == 0)
                        ret[0].sample_time += omp_get_wtime() - t_sample_s;
                }
            }
            double t_coo_s = omp_get_wtime();
            ret[ret.size() - 1 - i].ts.insert(ret[ret.size() - 1 - i].ts.end(),
                                              root_ts->begin(), root_ts->end());
            ret[ret.size() - 1 - i].nodes.insert(ret[ret.size() - 1 - i].nodes.end(),
                                                 root_nodes->begin(), root_nodes->end());
            ret[ret.size() - 1 - i].dts.resize(root_nodes->size());
            combine_coo(ret[ret.size() - 1 - i], _row, _col, _eid, _ts, _dts, _nodes, _out_node);
            ret[0].coo_time += omp_get_wtime() - t_coo_s;
        }
        ret[0].tot_time += omp_get_wtime() - t_s;
    }

    void sample(std::vector<NodeIDType> &root_nodes, std::vector<TimeStampType> &root_ts)
    {
        // a weird bug, dgl library seems to modify the total number of threads
        omp_set_num_threads(num_threads);
        ret.resize(0);
        bool first_layer = true;
        bool use_ptr = false;
        for (int i = 0; i < num_layers; i++)
        {
            ret.resize(ret.size() + num_history);
            if ((first_layer) || ((prop_time) && num_history == 1) || (recent))
            {
                first_layer = false;
                use_ptr = true;
            }
            else
                use_ptr = false;
            if (i == 0)
                sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, true);
            else
                sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, false);
        }
    }
};

template <typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory
    // see https://github.com/pybind/pybind11/issues/1042
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

PYBIND11_MODULE(sampler_core, m)
{
    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<std::vector<NodeIDType> &, std::vector<NodeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      std::vector<TimeStampType> &, std::vector<NodeIDType> &,
                      NodeIDType, NodeIDType>())
        .def("row", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.eid); })
        .def("ts", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.ts); })
        .def("dts", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.dts); })
        .def("nodes", [](const TemporalGraphBlock &tgb)
             { return vec2npy(tgb.nodes); })
        .def("dim_in", [](const TemporalGraphBlock &tgb)
             { return tgb.dim_in; })
        .def("dim_out", [](const TemporalGraphBlock &tgb)
             { return tgb.dim_out; })
        .def("tot_time", [](const TemporalGraphBlock &tgb)
             { return tgb.tot_time; })
        .def("ptr_time", [](const TemporalGraphBlock &tgb)
             { return tgb.ptr_time; })
        .def("search_time", [](const TemporalGraphBlock &tgb)
             { return tgb.search_time; })
        .def("sample_time", [](const TemporalGraphBlock &tgb)
             { return tgb.sample_time; })
        .def("coo_time", [](const TemporalGraphBlock &tgb)
             { return tgb.coo_time; });
             
    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<std::vector<EdgeIDType> &, std::vector<EdgeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      int, int, int, std::vector<int> &, bool, bool,
                      int, TimeStampType>())
        .def("sample", &ParallelSampler::sample)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler &ps)
             { return ps.ret; });
}