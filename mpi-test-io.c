#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File fh;
    MPI_Offset file_size;

    MPI_File_open(MPI_COMM_WORLD, "config.ini", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &file_size);

    // Allocate buffer to hold file content
    char *buffer = new char[file_size + 1];
    MPI_File_read_at_all(fh, 0, buffer, file_size, MPI_CHAR, MPI_STATUS_IGNORE);
    buffer[file_size] = '\0'; // Null-terminate

    MPI_File_close(&fh);

    std::map<std::string, std::string> config_map;

    // Only one process parses and prints
    if (rank == 0)
    {
        std::istringstream stream(buffer);
        std::string line;
        while (std::getline(stream, line))
        {
            if (line.empty() || line[0] == '#')
                continue; // Skip comments and blank lines
            auto delim_pos = line.find('=');
            if (delim_pos != std::string::npos)
            {
                std::string key = line.substr(0, delim_pos);
                std::string value = line.substr(delim_pos + 1);
                config_map[key] = value;
            }
        }

        // Print map content
        for (const auto &kv : config_map)
        {
            std::cout << kv.first << " = " << kv.second << std::endl;
        }
    }

    delete[] buffer;
    MPI_Finalize();
    return 0;
}
