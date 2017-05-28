// ==============================================================================
//  KratosShapeOptimizationApplication
//
//  License:         BSD License
//                   license: ShapeOptimizationApplication/license.txt
//
//  Main authors:    Baumgärtner Daniel, https://github.com/dbaumgaertner
//
// ==============================================================================

#ifndef MAPPER_VERTEX_MORPHING_MATRIX_FREE_H
#define MAPPER_VERTEX_MORPHING_MATRIX_FREE_H

// ------------------------------------------------------------------------------
// System includes
// ------------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <algorithm>

// ------------------------------------------------------------------------------
// External includes
// ------------------------------------------------------------------------------
#include <boost/python.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

// ------------------------------------------------------------------------------
// Project includes
// ------------------------------------------------------------------------------
#include "includes/define.h"
#include "processes/process.h"
#include "includes/node.h"
#include "includes/element.h"
#include "includes/model_part.h"
#include "includes/kratos_flags.h"
#include "spatial_containers/spatial_containers.h"
#include "utilities/timer.h"
#include "processes/node_erase_process.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/normal_calculation_utils.h"
#include "spaces/ublas_space.h"
#include "shape_optimization_application.h"
#include "filter_function.h"

// ==============================================================================

namespace Kratos
{

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

/// Short class definition.
/** Detail class definition.

*/

class MapperVertexMorphingMatrixFree
{
public:
    ///@name Type Definitions
    ///@{

    // Type definitions for better reading later
    typedef array_1d<double,3> array_3d;
    typedef Node < 3 > NodeType;
    typedef Node < 3 > ::Pointer NodeTypePointer;
    typedef std::vector<NodeType::Pointer> NodeVector;
    typedef std::vector<NodeType::Pointer>::iterator NodeIterator;
    typedef std::vector<double>::iterator DoubleVectorIterator;
    typedef ModelPart::ConditionsContainerType ConditionsArrayType;

    // Type definitions for tree-search
    typedef Bucket< 3, NodeType, NodeVector, NodeTypePointer, NodeIterator, DoubleVectorIterator > BucketType;
    typedef Tree< KDTreePartition<BucketType> > KDTree;    

    /// Pointer definition of MapperVertexMorphingMatrixFree
    KRATOS_CLASS_POINTER_DEFINITION(MapperVertexMorphingMatrixFree);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    MapperVertexMorphingMatrixFree( ModelPart& designSurface, Parameters& optimizationSettings )
        : mrDesignSurface( designSurface ),
          mNumberOfDesignVariables( designSurface.Nodes().size() ),
          mFilterType( optimizationSettings["design_variables"]["filter"]["filter_function_type"].GetString() ),
          mFilterRadius( optimizationSettings["design_variables"]["filter"]["filter_radius"].GetDouble() ),
          mMaxNumberOfNeighbors( optimizationSettings["design_variables"]["filter"]["max_nodes_in_filter_radius"].GetInt() )
    {
        CreateListOfNodesOfDesignSurface();
        CreateSearchTreeWithAllNodesOnDesignSurface();
        CreateFilterFunction();
        InitializeMappingVariables();
        AssignMappingIds();
    }

    /// Destructor.
    virtual ~MapperVertexMorphingMatrixFree()
    {
    }


    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    // ==============================================================================
    void CreateListOfNodesOfDesignSurface()
    {
        mListOfNodesOfDesignSurface.resize(mNumberOfDesignVariables);
        std::size_t counter = 0;
        for (ModelPart::NodesContainerType::iterator node_it = mrDesignSurface.NodesBegin(); node_it != mrDesignSurface.NodesEnd(); ++node_it)
        {
            NodeTypePointer pnode = *(node_it.base());
            mListOfNodesOfDesignSurface[counter++] = pnode;
        }
    }

    // --------------------------------------------------------------------------
    void CreateSearchTreeWithAllNodesOnDesignSurface()
    {
        boost::timer timer;        
        std::cout << "> Creating search tree to perform mapping..." << std::endl;        
        mpSearchTree = boost::shared_ptr<KDTree>(new KDTree(mListOfNodesOfDesignSurface.begin(), mListOfNodesOfDesignSurface.end(), mBucketSize));
        std::cout << "> Search tree created in: " << timer.elapsed() << " s" << std::endl;        
    }   

    // --------------------------------------------------------------------------
    void CreateFilterFunction()
    {
        mpFilterFunction = boost::shared_ptr<FilterFunction>(new FilterFunction(mFilterType, mFilterRadius));
    }     

    // --------------------------------------------------------------------------
    void InitializeMappingVariables()
    {
        x_variables_in_design_space.resize(mNumberOfDesignVariables,0.0);
        y_variables_in_design_space.resize(mNumberOfDesignVariables,0.0);
        z_variables_in_design_space.resize(mNumberOfDesignVariables,0.0);
        x_variables_in_geometry_space.resize(mNumberOfDesignVariables,0.0);
        y_variables_in_geometry_space.resize(mNumberOfDesignVariables,0.0);
        z_variables_in_geometry_space.resize(mNumberOfDesignVariables,0.0);
    }

    // --------------------------------------------------------------------------
    void AssignMappingIds()
    {
        unsigned int i = 0;
        for(auto& node_i : mrDesignSurface.Nodes())
            node_i.SetValue(MAPPING_ID,i++);
    }

    // --------------------------------------------------------------------------
    void MapToDesignSpace( const Variable<array_3d> &rNodalVariable, const Variable<array_3d> &rNodalVariableInDesignSpace )
    {
        boost::timer mapping_time;
        std::cout << "\n> Starting to map " << rNodalVariable.Name() << " to design space..." << std::endl;    
        
        RecreateSearchTreeIfGeometryHasChanged();
        ClearVectorsForMappingToDesignSpace();
        MapVariableComponentwiseToDesignSpace( rNodalVariable );
        AssignResultingDesignVectorsToNodalVariable( rNodalVariableInDesignSpace );

        std::cout << "> Time needed for mapping: " << mapping_time.elapsed() << " s" << std::endl;
    }

    // --------------------------------------------------------------------------
    void MapToGeometrySpace( const Variable<array_3d> &rNodalVariable, const Variable<array_3d> &rNodalVariableInGeometrySpace )
    {
        boost::timer mapping_time;
        std::cout << "\n> Starting to map " << rNodalVariable.Name() << " to geometry space..." << std::endl;

        RecreateSearchTreeIfGeometryHasChanged();
        ClearVectorsForMappingToGeometrySpace();
        MapVariableComponentwiseToGeometrySpace( rNodalVariable );
        AssignResultingGeometryVectorsToNodalVariable( rNodalVariableInGeometrySpace );

        std::cout << "> Time needed for mapping: " << mapping_time.elapsed() << " s" << std::endl;
    }

    // --------------------------------------------------------------------------
    void RecreateSearchTreeIfGeometryHasChanged()
    {
        if(HasGeometryChanged())
        {
            mpSearchTree.reset();
            CreateSearchTreeWithAllNodesOnDesignSurface();
        }
    }

    // --------------------------------------------------------------------------
    void ClearVectorsForMappingToDesignSpace()
    {
        x_variables_in_design_space.clear();
        y_variables_in_design_space.clear();
        z_variables_in_design_space.clear();
    }   

    // --------------------------------------------------------------------------
    void ClearVectorsForMappingToGeometrySpace()
    {
        x_variables_in_geometry_space.clear();
        y_variables_in_geometry_space.clear();
        z_variables_in_geometry_space.clear();
    }

    // --------------------------------------------------------------------------
    void MapVariableComponentwiseToDesignSpace( const Variable<array_3d>& rNodalVariable )
    {
        for(auto& node_i : mrDesignSurface.Nodes())
        {
            NodeVector neighbor_nodes( mMaxNumberOfNeighbors );
            std::vector<double> resulting_squared_distances( mMaxNumberOfNeighbors );
            unsigned int number_of_neighbors = mpSearchTree->SearchInRadius( node_i,
                                                                             mFilterRadius, 
                                                                             neighbor_nodes.begin(),
                                                                             resulting_squared_distances.begin(), 
                                                                             mMaxNumberOfNeighbors );

            std::vector<double> list_of_weights( number_of_neighbors, 0.0 );
            double sum_of_weights = 0.0;

            ThrowWarningIfMaxNodeNeighborsReached( node_i, number_of_neighbors );                                                                               
            ComputeWeightForAllNeighbors( node_i, neighbor_nodes, number_of_neighbors, list_of_weights, sum_of_weights );
            PerformLocalTransposeMapping( rNodalVariable, node_i, neighbor_nodes, number_of_neighbors, list_of_weights, sum_of_weights );            
        } 
    }

    // --------------------------------------------------------------------------
    void MapVariableComponentwiseToGeometrySpace( const Variable<array_3d>& rNodalVariable )
    {
        for(auto& node_i : mrDesignSurface.Nodes())
        {
            NodeVector neighbor_nodes(mMaxNumberOfNeighbors);
            std::vector<double> resulting_squared_distances(mMaxNumberOfNeighbors);
            unsigned int number_of_neighbors = mpSearchTree->SearchInRadius( node_i,
                                                                             mFilterRadius, 
                                                                             neighbor_nodes.begin(),
                                                                             resulting_squared_distances.begin(), 
                                                                             mMaxNumberOfNeighbors );

            std::vector<double> list_of_weights( number_of_neighbors, 0.0 );
            double sum_of_weights = 0.0;

            ThrowWarningIfMaxNodeNeighborsReached( node_i, number_of_neighbors );                                                                               
            ComputeWeightForAllNeighbors( node_i, neighbor_nodes, number_of_neighbors, list_of_weights, sum_of_weights );
            PerformLocalMapping( rNodalVariable, node_i, neighbor_nodes, number_of_neighbors, list_of_weights, sum_of_weights );
        }        
    }       

    // --------------------------------------------------------------------------
    void AssignResultingDesignVectorsToNodalVariable( const Variable<array_3d> &rNodalVariable )
    {
        for (ModelPart::NodeIterator node_i = mrDesignSurface.NodesBegin(); node_i != mrDesignSurface.NodesEnd(); ++node_i)
        {
            int i = node_i->GetValue(MAPPING_ID);

            Vector node_vector = ZeroVector(3);
            node_vector(0) = x_variables_in_design_space[i];
            node_vector(1) = y_variables_in_design_space[i];
            node_vector(2) = z_variables_in_design_space[i];
            node_i->FastGetSolutionStepValue(rNodalVariable) = node_vector;
        }
    }

    // --------------------------------------------------------------------------
    void AssignResultingGeometryVectorsToNodalVariable( const Variable<array_3d> &rNodalVariable )
    {
        for (ModelPart::NodeIterator node_i = mrDesignSurface.NodesBegin(); node_i != mrDesignSurface.NodesEnd(); ++node_i)
        {
            int i = node_i->GetValue(MAPPING_ID);

            Vector node_vector = ZeroVector(3);
            node_vector(0) = x_variables_in_geometry_space[i];
            node_vector(1) = y_variables_in_geometry_space[i];
            node_vector(2) = z_variables_in_geometry_space[i];
            node_i->FastGetSolutionStepValue(rNodalVariable) = node_vector;
        }
    } 

    // --------------------------------------------------------------------------
    void ThrowWarningIfMaxNodeNeighborsReached( ModelPart::NodeType& given_node, unsigned int number_of_neighbors )
    {
        if(number_of_neighbors >= mMaxNumberOfNeighbors)
            std::cout << "\n> WARNING!!!!! For node " << given_node.Id() << " and specified filter radius, maximum number of neighbor nodes (=" << mMaxNumberOfNeighbors << " nodes) reached!" << std::endl;
    }

    // --------------------------------------------------------------------------
    void ComputeWeightForAllNeighbors(  ModelPart::NodeType& design_node, 
                                        NodeVector& neighbor_nodes, 
                                        unsigned int number_of_neighbors,
                                        std::vector<double>& list_of_weights, 
                                        double& sum_of_weights )
    {
        for(unsigned int neighbor_itr = 0 ; neighbor_itr<number_of_neighbors ; neighbor_itr++)
        {
            ModelPart::NodeType& neighbor_node = *neighbor_nodes[neighbor_itr];
            double weight = mpFilterFunction->compute_weight( design_node.Coordinates(), neighbor_node.Coordinates() );

            list_of_weights[neighbor_itr] = weight;
            sum_of_weights += weight;
        }
    }

    // --------------------------------------------------------------------------
    void PerformLocalTransposeMapping( const Variable<array_3d>& rNodalVariable,
                                       ModelPart::NodeType& design_node, 
                                       NodeVector& neighbor_nodes, 
                                       unsigned int number_of_neighbors,
                                       std::vector<double>& list_of_weights, 
                                       double& sum_of_weights )
    {
        array_3d& nodal_variable = design_node.FastGetSolutionStepValue(rNodalVariable);
        for(unsigned int neighbor_itr = 0 ; neighbor_itr<number_of_neighbors ; neighbor_itr++)
        {
            ModelPart::NodeType& neighbor_node = *neighbor_nodes[neighbor_itr];
            int neighbor_node_mapping_id = neighbor_node.GetValue(MAPPING_ID);

            double weight = list_of_weights[neighbor_itr] / sum_of_weights;

            x_variables_in_design_space[neighbor_node_mapping_id] += weight*nodal_variable[0];
            y_variables_in_design_space[neighbor_node_mapping_id] += weight*nodal_variable[1];
            z_variables_in_design_space[neighbor_node_mapping_id] += weight*nodal_variable[2];
        }
    }

    // --------------------------------------------------------------------------
    void PerformLocalMapping( const Variable<array_3d>& rNodalVariable,
                              ModelPart::NodeType& design_node, 
                              NodeVector& neighbor_nodes, 
                              unsigned int number_of_neighbors,
                              std::vector<double>& list_of_weights, 
                              double& sum_of_weights )
    {
        int design_node_mapping_id = design_node.GetValue(MAPPING_ID);
        for(unsigned int neighbor_itr = 0 ; neighbor_itr<number_of_neighbors ; neighbor_itr++)
        {
            double weight = list_of_weights[neighbor_itr] / sum_of_weights;

            ModelPart::NodeType& node_j = *neighbor_nodes[neighbor_itr];
            array_3d& nodal_variable = node_j.FastGetSolutionStepValue(rNodalVariable);

            x_variables_in_geometry_space[design_node_mapping_id] += weight*nodal_variable[0];
            y_variables_in_geometry_space[design_node_mapping_id] += weight*nodal_variable[1];
            z_variables_in_geometry_space[design_node_mapping_id] += weight*nodal_variable[2];
        }
    }

    // --------------------------------------------------------------------------
    bool HasGeometryChanged()
    {
        double sumOfAllCoordinates = 0.0;
        for(auto& node_i : mrDesignSurface.Nodes())
        {
            array_3d& coord = node_i.Coordinates();
            sumOfAllCoordinates += coord[0] + coord[1] + coord[2];
        }

        if(IsFirstMappingOperation())
        {
            mControlSum = sumOfAllCoordinates;
            return false;
        }
        else if (mControlSum == sumOfAllCoordinates)
            return false;
        else 
        {
            mControlSum = sumOfAllCoordinates;
            return true;
        }
    } 

    // --------------------------------------------------------------------------
    bool IsFirstMappingOperation()
    {
        if(mControlSum == 0.0)
            return true;
        else 
             return false;
    }   

    // ==============================================================================

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
        return "MapperVertexMorphingMatrixFree";
    }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "MapperVertexMorphingMatrixFree";
    }

    /// Print object's data.
    virtual void PrintData(std::ostream& rOStream) const
    {
    }


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

    // ==============================================================================
    // Initialized by class constructor
    // ==============================================================================
    ModelPart& mrDesignSurface;
    const unsigned int mNumberOfDesignVariables;
    std::string mFilterType;
    double mFilterRadius;
    unsigned int mMaxNumberOfNeighbors;            
    FilterFunction::Pointer mpFilterFunction;

    // ==============================================================================
    // Variables for spatial search
    // ==============================================================================
    unsigned int mBucketSize = 100;
    NodeVector mListOfNodesOfDesignSurface;
    KDTree::Pointer mpSearchTree;

    // ==============================================================================
    // Variables for mapping
    // ==============================================================================
    Vector x_variables_in_design_space, y_variables_in_design_space, z_variables_in_design_space;    
    Vector x_variables_in_geometry_space, y_variables_in_geometry_space, z_variables_in_geometry_space;
    double mControlSum = 0.0;


    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{


    ///@}
    ///@name Private  Access
    ///@{


    ///@}
    ///@name Private Inquiry
    ///@{


    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
//      MapperVertexMorphingMatrixFree& operator=(MapperVertexMorphingMatrixFree const& rOther);

    /// Copy constructor.
//      MapperVertexMorphingMatrixFree(MapperVertexMorphingMatrixFree const& rOther);


    ///@}

}; // Class MapperVertexMorphingMatrixFree

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

///@}


}  // namespace Kratos.

#endif // MAPPER_VERTEX_MORPHING_MATRIX_FREE_H