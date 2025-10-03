#pragma once
#include <Cabana_Grid.hpp>
#include "outputImpl.hpp"
/*
accept any number of cabana grid array,
select a writer then write all of them
*/
namespace CabanaDSMC{
template<class WriterType, class... ArrayTypes>
void writeFields(
    const std::string& file_directory,
    const int& time_step,
    const double& time,
    const std::shared_ptr<WriterType>& writer,
    const std::shared_ptr<ArrayTypes>&... arrays
)
{
    (writer->write(file_directory, time_step, time, arrays), ...);
}



struct WriteBasic {

};

struct WriteAddtional {

};

template<class WriterType, class CellDataType>
void write(
    const WriteBasic&,
    const std::string& file_directory,
    const int& time_step,
    const double& time,
    const std::shared_ptr<WriterType>& writer,
    const std::shared_ptr<CellDataType>& cell_data
    )
{
    writeFields(
        file_directory,
        time_step,
        time,
        writer,
        cell_data->Velocity,
        cell_data->Density,
        cell_data->Temperature,
        cell_data->Pressure);
}
}