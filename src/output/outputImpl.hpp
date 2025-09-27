#pragma once
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
namespace CabanaDSMC{
template<class Derived, class ArrayType>
class Writer{
public:
    using array_t = ArrayType;
    Writer() = default;
    void write(
        const std::string& file_directory,
        const int& time_step,
        const double& time,
        const std::shared_ptr<array_t>& array
    )
    {
        static_cast<Derived*>(this)->writeImpl(file_directory, time_step, time, array);
    }
};

template<class ArrayType>
class BovWriter : public Writer<BovWriter<ArrayType>, ArrayType>
{
public:
    using base_type = Writer<BovWriter<ArrayType>, ArrayType>;
    using array_t = ArrayType;
    BovWriter() = default;
    void writeImpl(
        const std::string& file_directory,
        const int& time_step,
        const double& time,
        const std::shared_ptr<array_t>& array
    )
    {
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            file_directory,
            time_step,
            time,
            *array
        );
    }
};
}