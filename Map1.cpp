#include <mpi.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Function to perform local sorting
void local_sort(std::vector<int>& data) {
    std::sort(data.begin(), data.end());
}

// Function to find splitters using parallel selection
std::vector<int> find_splitters(const std::vector<int>& data, int p) {
    int n = data.size();
    std::vector<int> splitters(p - 1);
    for (int i = 1; i < p; ++i) {
        splitters[i - 1] = data[i * n / p];
    }
    return splitters;
}

// Function to determine target processor for an element
int determine_target_processor(int element, const std::vector<int>& splitters) {
    for (size_t i = 0; i < splitters.size(); ++i) {
        if (element <= splitters[i]) {
            return i;
        }
    }
    return splitters.size();
}

// Function to merge two sorted lists
std::vector<int> merge_two_lists(const std::vector<int>& list1, const std::vector<int>& list2) {
    std::vector<int> result;
    result.reserve(list1.size() + list2.size());
    std::merge(list1.begin(), list1.end(), list2.begin(), list2.end(), std::back_inserter(result));
    return result;
}

// Function to perform parallel sorting
void parallel_sort(std::vector<int>& local_data, int p) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Step 1: Local sort
    local_sort(local_data);

    // Step 2: Find splitters
    std::vector<int> splitters;
    if (rank == 0) {
        std::vector<int> all_data;
        for (int i = 0; i < p; ++i) {
            int local_size;
            MPI_Recv(&local_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> temp(local_size);
            MPI_Recv(temp.data(), local_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            all_data.insert(all_data.end(), temp.begin(), temp.end());
        }
        std::sort(all_data.begin(), all_data.end());
        splitters = find_splitters(all_data, p);
        for (int i = 0; i < p; ++i) {
            MPI_Send(splitters.data(), splitters.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        int local_size = local_data.size();
        MPI_Send(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        splitters.resize(p - 1);
        MPI_Recv(splitters.data(), splitters.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 3: Element routing
    std::vector<std::vector<int>> buckets(p);
    for (const int& element : local_data) {
        int target_processor = determine_target_processor(element, splitters);
        buckets[target_processor].push_back(element);
    }

    // Step 4: Send and receive data
    std::vector<int> merged_data;
    for (int i = 0; i < p; ++i) {
        if (i != rank) {
            int send_size = buckets[i].size();
            MPI_Send(&send_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(buckets[i].data(), send_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < p; ++i) {
        if (i != rank) {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> temp(recv_size);
            MPI_Recv(temp.data(), recv_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merged_data = merge_two_lists(merged_data, temp);
        } else {
            merged_data = merge_two_lists(merged_data, buckets[i]);
        }
    }

    // Step 5: Final merge using binary tree-based merging
    int step = 1;
    while (step < p) {
        if (rank % (2 * step) == 0) {
            if (rank + step < p) {
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> temp(recv_size);
                MPI_Recv(temp.data(), recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                merged_data = merge_two_lists(merged_data, temp);
            }
        } else {
            int target = rank - step;
            int send_size = merged_data.size();
            MPI_Send(&send_size, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            MPI_Send(merged_data.data(), send_size, MPI_INT, target, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    // Output the sorted data on rank 0
    if (rank == 0) {
        for (const int& val : merged_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::srand(std::time(0) + rank);

    // Generate random data
    int n = 100; // Number of elements per processor
    std::vector<int> local_data(n);
    std::generate(local_data.begin(), local_data.end(), std::rand);

    parallel_sort(local_data, size);

    MPI_Finalize();
    return 0;
}
