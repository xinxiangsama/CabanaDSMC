#pragma once
#include <Cabana_Grid.hpp>

namespace CabanaDSMC{
namespace Particle{

namespace Field{
    struct Position : Cabana::Field::Vector<double, 3> {static std::string label() { return "position"; }};

    struct Velocity : Cabana::Field::Vector<double, 3> {static std::string label() { return "velocity"; }};

    struct RotEnergy : Cabana::Field::Scalar<double> {static std::string label() { return "rotational energy"; }};

    struct VibEnergy : Cabana::Field::Scalar<double> {static std::string label() { return "vibrational energy"; }};

    struct SpeciesID : Cabana::Field::Scalar<uint32_t> {static std::string label() { return "species id"; }};

    struct GlobalID : Cabana::Field::Scalar<std::size_t> {static std::string label() { return "global id"; }};

    struct CellID : Cabana::Field::Vector<u_int32_t, 3> {static std::string label() { return "cell id"; }};

    struct IsActive : Cabana::Field::Scalar<bool> {static std::string label() { return "if valid"; }};
}

    template<class MemorySpace, uint32_t VectorLength = 16>
    using GridParticleList = Cabana::Grid::ParticleList<MemorySpace, VectorLength,
                                                    Field::Position,
                                                    Field::Velocity,
                                                    Field::RotEnergy,
                                                    Field::VibEnergy,
                                                    Field::SpeciesID,
                                                    Field::GlobalID,
                                                    Field::CellID,
                                                    Field::IsActive>;
    template<class MemorySpace, uint32_t VectorLength = 16>
    using ParticleList = Cabana::ParticleList<MemorySpace, VectorLength,
                                                    Field::Position,
                                                    Field::Velocity,
                                                    Field::RotEnergy,
                                                    Field::VibEnergy,
                                                    Field::SpeciesID,
                                                    Field::GlobalID,
                                                    Field::CellID,
                                                    Field::IsActive>;
    struct Species{
        double mass;
        double diameter;
        double omega;
        double Tref;
        uint32_t Zrot;
    };

    template<class MemorySpace>
    using SpeciesList = Kokkos::View<Species*, MemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
}

}