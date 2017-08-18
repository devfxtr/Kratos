//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Philipp Bucher, Jordi Cotela
//
// See Master-Thesis P.Bucher
// "Development and Implementation of a Parallel
//  Framework for Non-Matching Grid Mapping"

#if !defined(KRATOS_MAPPER_FACTORY_NEW_H_INCLUDED )
#define  KRATOS_MAPPER_FACTORY_NEW_H_INCLUDED

// System includes
#include "linear_solvers/linear_solver.h"

#include "spaces/ublas_space.h" // Always needed, for "LocalSpaceType"

#ifdef KRATOS_USING_MPI // mpi-parallel compilation
#include "trilinos_space.h"
#include "Epetra_FEVector.h"
#endif

// External includes


// Project includes
#include "includes/define.h"
#include "includes/kratos_parameters.h"

#include "custom_mappers/nearest_neighbor_mapper.h"
#include "custom_mappers/nearest_element_mapper.h"
#include "custom_mappers/nearest_neighbor_mapper_matrix.h"
#include "custom_mappers/mortar_mapper.h"


namespace Kratos
{
///@addtogroup ApplicationNameApplication
///@{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/// Python Interface of the MappingApplication
/** This class is a proper factory and returns the a mapper // TODO do this properly
*/
class MapperFactoryNew // TODO change the name
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of MapperFactoryNew
    KRATOS_CLASS_POINTER_DEFINITION(MapperFactoryNew);

    typedef UblasSpace<double, CompressedMatrix, Vector> SerialSparseSpaceType;
    typedef UblasSpace<double, Matrix, Vector> LocalSpaceType;
    typedef LinearSolver<SerialSparseSpaceType, LocalSpaceType> SerialLinearSolverType; // for Mortar
    
    // Overwrite the SparseSpaceType in case of mpi-parallel execution
    #ifdef KRATOS_USING_MPI // mpi-parallel compilation
    typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;
    typedef LinearSolver<TrilinosSparseSpaceType, LocalSpaceType> TrilinosLinearSolverType; // for Mortar
    #endif

    ///@}
    ///@name Life Cycle
    ///@{

    
    /// Destructor.
    virtual ~MapperFactoryNew() { }


    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    // void ReadInterfaceModelParts()
    // {
    //     int echo_level = 0;
    //     // read the echo_level temporarily, bcs the mJsonParameters have not yet been validated and defaults assigned
    //     if (mJsonParameters.Has("echo_level"))
    //     {
    //         echo_level = std::max(echo_level, mJsonParameters["echo_level"].GetInt());
    //     }

    //     int comm_rank_origin = mrModelPartOrigin.GetCommunicator().MyPID();
    //     int comm_rank_destination = mrModelPartDestination.GetCommunicator().MyPID();

    //     if (mJsonParameters.Has("interface_submodel_part_origin"))
    //     {
    //         std::string name_interface_submodel_part = mJsonParameters["interface_submodel_part_origin"].GetString();
    //         mpInterfaceModelPartOrigin = &mrModelPartOrigin.GetSubModelPart(name_interface_submodel_part);

    //         if (echo_level >= 3 && comm_rank_origin == 0)
    //         {
    //             std::cout << "Mapper: SubModelPart used for Origin-ModelPart" << std::endl;
    //         }
    //     }
    //     else
    //     {
    //         mpInterfaceModelPartOrigin = &mrModelPartOrigin;

    //         if (echo_level >= 3 && comm_rank_origin == 0)
    //         {
    //             std::cout << "Mapper: Main ModelPart used for Origin-ModelPart" << std::endl;
    //         }
    //     }

    //     if (mJsonParameters.Has("interface_submodel_part_destination"))
    //     {
    //         std::string name_interface_submodel_part = mJsonParameters["interface_submodel_part_destination"].GetString();
    //         mpInterfaceModelPartDestination = &mrModelPartDestination.GetSubModelPart(name_interface_submodel_part);

    //         if (echo_level >= 3 && comm_rank_destination == 0)
    //         {
    //             std::cout << "Mapper: SubModelPart used for Destination-ModelPart" << std::endl;
    //         }
    //     }
    //     else
    //     {
    //         mpInterfaceModelPartDestination = &mrModelPartDestination;

    //         if (echo_level >= 3 && comm_rank_destination == 0)
    //         {
    //             std::cout << "Mapper: Main ModelPart used for Destination-ModelPart" << std::endl;
    //         }
    //     }
    // }

    static Mapper* CreateMapper(ModelPart& rModelPartOrigin, 
                                        ModelPart& rModelPartDestination,
                                        Parameters JsonParameters)
    {
        Mapper* mapper;
        // double start_time = MapperUtilities::GetCurrentTime();

        if (!JsonParameters.Has("mapper_type"))
        {
            KRATOS_ERROR << "No \"mapper_type\" defined in json" << std::endl;
        }

        const std::string mapper_type = JsonParameters["mapper_type"].GetString();

        if (mapper_type == "NearestNeighbor")
        {
            if (JsonParameters.Has("approximation_tolerance"))
            {
                KRATOS_ERROR << "Invalid Parameter \"approximation_tolerance\" "
                             << "specified for Nearest Neighbor Mapper" << std::endl;
            }

            mapper = new NearestNeighborMapper(rModelPartOrigin,
                                       rModelPartDestination,
                                       JsonParameters);
        }
        else if (mapper_type == "NearestElement")
        {
            mapper = new NearestElementMapper(rModelPartOrigin,
                                       rModelPartDestination,
                                       JsonParameters);

        } 
        else if (mapper_type == "NearestNeighborMatrixBased") 
        {
#ifdef KRATOS_USING_MPI // mpi-parallel compilation
            if (MapperUtilities::TotalProcesses() > 1) // parallel execution, i.e. mpi imported in python
            {
                mapper = new NearestNeighborMapperMatrix<
                    TrilinosSparseSpaceType, LocalSpaceType, TrilinosLinearSolverType>(
                        rModelPartOrigin,
                        rModelPartDestination,
                        JsonParameters);
            }
            else
            {
                mapper = new NearestNeighborMapperMatrix<
                    SerialSparseSpaceType, LocalSpaceType, SerialLinearSolverType>(
                        rModelPartOrigin,
                        rModelPartDestination,
                        JsonParameters);
            }

#else
            mapper = new NearestNeighborMapperMatrix<
                SerialSparseSpaceType, LocalSpaceType, SerialLinearSolverType>(
                    rModelPartOrigin,
                    rModelPartDestination,
                    JsonParameters);
#endif
        } 
        /*else if (mapper_type == "NearestElementMatrixBased") {
              mapper = Mapper::Pointer(new RBFMapper(rModelPartOrigin,
                                                         rModelPartDestination,
                                                         JsonParameters));
        }*/
        else if (mapper_type == "Mortar") 
        {
#ifdef KRATOS_USING_MPI // mpi-parallel compilation
            KRATOS_WATCH(MapperUtilities::TotalProcesses())
            if (MapperUtilities::TotalProcesses() > 1) // parallel execution, i.e. mpi imported in python
            {
                mapper = new MortarMapper<
                    TrilinosSparseSpaceType, LocalSpaceType, TrilinosLinearSolverType>(
                        rModelPartOrigin,
                        rModelPartDestination,
                        JsonParameters);
                KRATOS_WATCH("MPI Mortar Mapper")
            }
            else
            {
                mapper = new MortarMapper<
                    SerialSparseSpaceType, LocalSpaceType, SerialLinearSolverType>(
                        rModelPartOrigin,
                        rModelPartDestination,
                        JsonParameters);
                KRATOS_WATCH("Serial Mortar Mapper")
                        }
#else
            mapper = new MortarMapper<
                SerialSparseSpaceType, LocalSpaceType, SerialLinearSolverType>(
                    rModelPartOrigin,
                    rModelPartDestination,
                    JsonParameters);
#endif
        } 
        /*else if (mapper_type == "IGA") {
              mapper = Mapper::Pointer(new IGAMapper(*rModelPartOrigin,
                                                         *rModelPartDestination,
                                                         mJsonParameters));

        } */
        else
        {
            KRATOS_ERROR << "Selected Mapper \"" << mapper_type << "\" not implemented" << std::endl;
        }

        // double elapsed_time = MapperUtilities::GetCurrentTime() - start_time;

        // mpMapper->pGetMapperCommunicator()->PrintTime(mapper_type,
        //         "Mapper Construction",
        //         elapsed_time);
        return mapper;
    }


    ///@}
    ///@name Access
    ///@{


    ///@}
    ///@name Inquiry
    ///@{


    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    virtual std::string Info() const
    {
        std::stringstream buffer;
        buffer << "MapperFactoryNew" ;
        return buffer.str();
    }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "MapperFactoryNew";
    }

    /// Print object's data.
    virtual void PrintData(std::ostream& rOStream) const {}


    ///@}
    ///@name Friends
    ///@{


    ///@}

protected:
    ///@name Protected static Member Variables
    ///@{


    ///@}
    ///@name Protected member Variables
    ///@{


    ///@}
    ///@name Protected Operators
    ///@{


    ///@}
    ///@name Protected Operations
    ///@{


    ///@}
    ///@name Protected  Access
    ///@{


    ///@}
    ///@name Protected Inquiry
    ///@{


    ///@}
    ///@name Protected LifeCycle
    ///@{


    ///@}

private:
    ///@name Static Member Variables
    ///@{


    ///@}
    ///@name Member Variables
    ///@{

    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{

  /// Default constructor.
  MapperFactoryNew()
  {
  }

  ///@}
  ///@name Private  Access
  ///@{

  ///@}
  ///@name Private Inquiry
  ///@{s

  ///@}
  ///@name Un accessible methods
  ///@{

  /// Assignment operator.
  MapperFactoryNew &operator=(MapperFactoryNew const &rOther);

  //   /// Copy constructor.
  //   MapperFactoryNew(MapperFactoryNew const& rOther){}

  ///@}

}; // Class MapperFactoryNew

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
inline std::istream& operator >> (std::istream& rIStream,
                                  MapperFactoryNew& rThis)
{
    return rIStream;
}

/// output stream function
inline std::ostream& operator << (std::ostream& rOStream,
                                  const MapperFactoryNew& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}
///@}

///@} addtogroup block

}  // namespace Kratos.

#endif // KRATOS_MAPPER_FACTORY_NEW_H_INCLUDED  defined