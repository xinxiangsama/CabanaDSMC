#pragma once
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
namespace CabanaDSMC{
template<class Derived>
class Writer{
public:
    Writer() = default;
    template<class ArrayType>
    void write(
        const std::string& file_directory,
        const int& time_step,
        const double& time,
        const std::shared_ptr<ArrayType>& array
    )
    {
        static_cast<Derived*>(this)->writeImpl(file_directory, time_step, time, array);
    }
};

class BovWriter : public Writer<BovWriter>
{
public:
    BovWriter() = default;
    template<class ArrayType>
    void writeImpl(
        const std::string& file_directory,
        const int& time_step,
        const double& time,
        const std::shared_ptr<ArrayType>& array
    )
    {   
        using exec_space = typename ArrayType::execution_space;
        auto field_name = array->label();
        std::string full_file_path = file_directory + field_name;
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            exec_space {},
            full_file_path,
            time_step,
            time,
            *array
        );
    }
};
}
