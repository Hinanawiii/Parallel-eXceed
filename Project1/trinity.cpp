#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Function prototypes for the sorting algorithms
void optimal_sort(std::vector<int>& local_data, int rank, int size);
void novel_parallel_sort(std::vector<int>& local_data, int rank, int size);
void viral_parallel_sort(std::vector<int>& local_data, int rank, int size);

// Function prototypes for data generation
void generate_data(const std::string& filename, size_t size_in_kb);

// Function to read data from a file
void read_data(const std::string& filename, std::vector<int>& data) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    infile.seekg(0, std::ios::end);
    size_t filesize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    data.resize(filesize / sizeof(int));
    infile.read(reinterpret_cast<char*>(data.data()), filesize);
    infile.close();
}

// Function to run a test for a given sorting algorithm
void run_test(const std::string& filename, void (*sort_func)(std::vector<int>&, int, int)) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> data;
    read_data(filename, data);

    size_t local_size = data.size() / size;
    std::vector<int> local_data(local_size);
    MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting the timer
    auto start = std::chrono::high_resolution_clock::now();

    sort_func(local_data, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize after finishing the sort
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double local_elapsed = elapsed.count();
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sorting time for " << filename << ": " << max_elapsed << " seconds" << std::endl;
    }

    MPI_Gather(local_data.data(), local_size, MPI_INT, data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Verify sorted data and print results
        if (std::is_sorted(data.begin(), data.end())) {
            std::cout << "Data sorted successfully by " << filename << std::endl;
        }
        else {
            std::cout << "Data not sorted by " << filename << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    std::vector<size_t> sizes_in_kb = { 64, 128, 256, 512, 1024, 2048, 4096 };
    for (size_t size : sizes_in_kb) {
        generate_data("data_" + std::to_string(size) + "KB.dat", size);
    }

    std::vector<std::string> filenames = {
        "data_64KB.dat",
        "data_128KB.dat",
        "data_256KB.dat",
        "data_512KB.dat",
        "data_1024KB.dat",
        "data_2048KB.dat",
        "data_4096KB.dat"
    };

    for (const auto& filename : filenames) {
        std::cout << "Testing with dataset: " << filename << std::endl;
        run_test(filename, optimal_sort);
        run_test(filename, novel_parallel_sort);
        run_test(filename, viral_parallel_sort);
    }

    MPI_Finalize();
    return 0;
}

// Data generation code
void generate_data(const std::string& filename, size_t size_in_kb) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        exit(1);
    }

    size_t size_in_bytes = size_in_kb * 1024;
    std::vector<int> data(size_in_bytes / sizeof(int));
    srand(time(0));

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = rand();
    }

    outfile.write(reinterpret_cast<char*>(data.data()), size_in_bytes);
    outfile.close();
}

// Optimal sorting algorithm
void optimal_sort(std::vector<int>& local_data, int rank, int size) {
    std::sort(local_data.begin(), local_data.end());

    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                int partner = rank + step;
                int recv_size;
                MPI_Status status;

                MPI_Recv(&recv_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &status);
                std::vector<int> partner_data(recv_size);
                MPI_Recv(partner_data.data(), recv_size, MPI_INT, partner, 0, MPI_COMM_WORLD, &status);

                std::vector<int> merged_data(local_data.size() + partner_data.size());
                std::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(), merged_data.begin());
                local_data = merged_data;
            }
        }
        else {
            int partner = rank - step;
            int send_size = local_data.size();

            MPI_Send(&send_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_data.data(), send_size, MPI_INT, partner, 0, MPI_COMM_WORLD);

            break;
        }
        step *= 2;
    }
}

// Novel parallel sorting algorithm
void novel_parallel_sort(std::vector<int>& local_data, int rank, int size) {
    local_sort(local_data);

    int num_splitters = size - 1;
    std::vector<int> splitters = find_splitters(local_data, num_splitters);

    redistribute_data(local_data, splitters, rank, size);

    local_sort(local_data);
}

void local_sort(std::vector<int>& data) {
    std::sort(data.begin(), data.end());
}

std::vector<int> find_splitters(const std::vector<int>& data, int num_splitters) {
    std::vector<int> splitters(num_splitters);
    int step = data.size() / (num_splitters + 1);
    for (int i = 0; i < num_splitters; ++i) {
        splitters[i] = data[(i + 1) * step];
    }
    return splitters;
}

void redistribute_data(std::vector<int>& local_data, const std::vector<int>& splitters, int rank, int size) {
    std::vector<int> send_counts(size, 0);
    std::vector<int> recv_counts(size);

    auto it = local_data.begin();
    for (int i = 0; i < size; ++i) {
        auto next_it = (i == size - 1) ? local_data.end() : std::upper_bound(it, local_data.end(), splitters[i]);
        send_counts[i] = std::distance(it, next_it);
        it = next_it;
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    std::partial_sum(send_counts.begin(), send_counts.end() - 1, send_displs.begin() + 1);
    std::partial_sum(recv_counts.begin(), recv_counts.end() - 1, recv_displs.begin() + 1);

    int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
    std::vector<int> recv_data(total_recv);

    MPI_Alltoallv(local_data.data(), send_counts.data(), send_displs.data(), MPI_INT,
        recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

    local_data = std::move(recv_data);
}

// Viral parallel sorting algorithm
void viral_parallel_sort(std::vector<int>& local_data, int rank, int size) {
    local_sort(local_data);

    int num_splitters = size - 1;
    std::vector<int> splitters = find_splitters(local_data, num_splitters);

    redistribute_data(local_data, splitters, rank, size);

    local_sort(local_data);
}
