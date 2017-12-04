// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
// 

// System includes

// External includes

// Project includes

/* Custom utilities */
#include "custom_utilities/contact_utilities.h"
#include "custom_utilities/tree_contact_search.h"

namespace Kratos
{
template<unsigned int TDim, unsigned int TNumNodes>
TreeContactSearch<TDim, TNumNodes>::TreeContactSearch( 
    ModelPart & rMainModelPart, 
    Parameters ThisParameters 
    ):mrMainModelPart(rMainModelPart), 
      mThisParameters(ThisParameters)
{        
    KRATOS_ERROR_IF(mrMainModelPart.HasSubModelPart("Contact") == false) << "WARNING:: Please add the Contact submodelpart to your modelpart list" << std::endl;
    
    Parameters DefaultParameters = Parameters(R"(
    {
        "allocation_size"                      : 1000, 
        "bucket_size"                          : 4, 
        "search_factor"                        : 2.0, 
        "type_search"                          : "InRadius", 
        "check_gap"                            : true, 
        "condition_name"                       : "",  
        "final_string"                         : "",  
        "inverted_search"                      : false
    })" );
    
    mThisParameters.ValidateAndAssignDefaults(DefaultParameters);
    
    mCheckGap = mThisParameters["check_gap"].GetBool();
    mInvertedSearch = mThisParameters["inverted_search"].GetBool();
    
    // Check if the computing contact submodelpart
    if (mrMainModelPart.HasSubModelPart("ComputingContact") == false) // We check if the submodelpart where the actual conditions used to compute contact are going to be computed 
    {
        mrMainModelPart.CreateSubModelPart("ComputingContact");
    }
    else // We clean the existing modelpart
    {
        ModelPart& computing_contact_model_part = mrMainModelPart.GetSubModelPart("ComputingContact"); 
        ConditionsArrayType& conditions_array = computing_contact_model_part.Conditions();
        const int num_conditions = static_cast<int>(conditions_array.size());

    #ifdef _OPENMP
        #pragma omp parallel for 
    #endif
        for(int i = 0; i < num_conditions; ++i) 
        {
            auto it_cond = conditions_array.begin() + i;
            it_cond->Set(TO_ERASE, true);
        }
        
        computing_contact_model_part.RemoveConditions(TO_ERASE);
    }
    
    // Updating the base condition
    GeometryType& auxiliar_geom = mrMainModelPart.GetSubModelPart("Contact").Conditions().begin()->GetGeometry();
    mGeometryType = auxiliar_geom.GetGeometryType();
    mConditionName = mThisParameters["condition_name"].GetString();
    if (mConditionName == "") 
        mCreateAuxiliarConditions = false;
    else 
    {
        mCreateAuxiliarConditions = true;
        mConditionName.append("Condition"); 
        mConditionName.append(std::to_string(TDim));
        mConditionName.append("D"); 
        mConditionName.append(std::to_string(auxiliar_geom.size())); 
        mConditionName.append("N"); 
        mConditionName.append(mThisParameters["final_string"].GetString());
    }
    
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        it_node->Set(ACTIVE, false);
    }
    
    // Iterate in the conditions
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());

#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        it_cond->Set(ACTIVE, false);
    }
}   
    
/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::InitializeMortarConditions()
{
//     // The allocation size
//     const unsigned int allocation_size = mThisParameters["allocation_size"].GetInt();
    
    // Iterate in the conditions
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());

#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;

        if (it_cond->Has(INDEX_MAP) == false) it_cond->SetValue(INDEX_MAP, IndexMap::Pointer(new IndexMap)); 
//             it_cond->GetValue(INDEX_MAP)->reserve(allocation_size); 
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::TotalClearScalarMortarConditions()
{
    TotalResetContactOperators();
    
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        it_node->FastGetSolutionStepValue(SCALAR_LAGRANGE_MULTIPLIER) = 0.0;
    }  
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::TotalClearComponentsMortarConditions()
{
    TotalResetContactOperators();
    
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        noalias(it_node->FastGetSolutionStepValue(VECTOR_LAGRANGE_MULTIPLIER)) = ZeroVector(3);
    }  
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::TotalClearALMFrictionlessMortarConditions()
{        
    TotalResetContactOperators();
    
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        it_node->FastGetSolutionStepValue(NORMAL_CONTACT_STRESS) = 0.0;
    }  
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::PartialClearScalarMortarConditions()
{
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        it_node->FastGetSolutionStepValue(SCALAR_LAGRANGE_MULTIPLIER) = 0.0;
    } 
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::PartialClearComponentsMortarConditions()
{
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        noalias(it_node->FastGetSolutionStepValue(VECTOR_LAGRANGE_MULTIPLIER)) = ZeroVector(3);
    } 
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::PartialClearALMFrictionlessMortarConditions()
{
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        it_node->FastGetSolutionStepValue(NORMAL_CONTACT_STRESS) = 0.0;
    } 
}
    
/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::CreatePointListMortar()
{
    // Clearing the vector
    mPointListDestination.clear();
    
    // Iterate in the conditions
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());

    // Creating a buffer for parallel vector fill
    const int num_threads = OpenMPUtils::GetNumThreads();
    std::vector<PointVector> points_buffer(num_threads);

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif
        const int thread_id = OpenMPUtils::ThisThread();

    #ifdef _OPENMP
        #pragma omp for
    #endif
        for(int i = 0; i < num_conditions; ++i) 
        {
            auto it_cond = conditions_array.begin() + i;
            
            if (it_cond->Is(MASTER) == !mInvertedSearch)
            {
                const PointTypePointer& p_point = PointTypePointer(new PointItem((*it_cond.base())));
                (points_buffer[thread_id]).push_back(p_point);
            }
        }
        
        // Combine buffers together
        #pragma omp single
        {
            for( auto& point_buffer : points_buffer)
            {
                std::move(point_buffer.begin(),point_buffer.end(),back_inserter(mPointListDestination));
            }
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef KRATOS_DEBUG
    // NOTE: We check the list
    for (unsigned int i_point = 0; i_point < mPointListDestination.size(); ++i_point )
    {
        mPointListDestination[i_point]->Check();
    }
//     std::cout << "The list is properly built" << std::endl;
#endif
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::UpdatePointListMortar()
{
    const double delta_time = mrMainModelPart.GetProcessInfo()[DELTA_TIME];
    
    const int num_points = static_cast<int>(mPointListDestination.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_points; ++i) mPointListDestination[i]->UpdatePoint(delta_time);
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::UpdateMortarConditions()
{        
    // We update the list of points
    UpdatePointListMortar();
    
    // Calculate the mean of the normal in all the nodes
    ContactUtilities::ComputeNodesMeanNormalModelPart(mrMainModelPart.GetSubModelPart("Contact")); 
    
    // We get the computing model part
    std::size_t condition_id = static_cast<std::size_t>(mrMainModelPart.Conditions().size());
    ModelPart& computing_contact_model_part = mrMainModelPart.GetSubModelPart("ComputingContact"); 
    
    const double delta_time = mrMainModelPart.GetProcessInfo()[DELTA_TIME];
    
    // We check if we are in a dynamic or static case
    const bool dynamic = mrMainModelPart.NodesBegin()->SolutionStepsDataHas(VELOCITY_X) ;
    
    // Some auxiliar values
    const unsigned int allocation_size = mThisParameters["allocation_size"].GetInt();           // Allocation size for the vectors and max number of potential results 
    const double search_factor = mThisParameters["search_factor"].GetDouble();                  // The search factor to be considered 
    SearchTreeType type_search = ConvertSearchTree(mThisParameters["type_search"].GetString()); // The search tree considered
    unsigned int bucket_size = mThisParameters["bucket_size"].GetInt();                         // Bucket size for kd-tree
    
    // Create a tree
    // It will use a copy of mNodeList (a std::vector which contains pointers)
    // Copying the list is required because the tree will reorder it for efficiency
    KDTree tree_points(mPointListDestination.begin(), mPointListDestination.end(), bucket_size);
    
    // Iterate in the conditions
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());

// #ifdef _OPENMP
//     #pragma omp parallel for firstprivate(tree_points) // FIXME: Make me parallel!!! 
// #endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        
        if (it_cond->Is(SLAVE) == !mInvertedSearch)
        {
            // Initialize values
            PointVector points_found(allocation_size);
            
            unsigned int number_points_found = 0;    
            
            if (type_search == KdtreeInRadius)
            {
                GeometryType& geometry = it_cond->GetGeometry();
                const Point& center = dynamic ? ContactUtilities::GetHalfJumpCenter(geometry, delta_time) : geometry.Center(); // NOTE: Center in half delta time or real center
                
                const double search_radius = search_factor * Radius(it_cond->GetGeometry());

                number_points_found = tree_points.SearchInRadius(center, search_radius, points_found.begin(), allocation_size);
            }
            else if (type_search == KdtreeInBox)
            {
                // Auxiliar values
                const double length_search = search_factor * it_cond->GetGeometry().Length();
                
                // Compute max/min points
                NodeType min_point, max_point;
                it_cond->GetGeometry().BoundingBox(min_point, max_point);
                
                // Get the normal in the extrema points
                Vector N_min, N_max;
                GeometryType::CoordinatesArrayType local_point_min, local_point_max;
                it_cond->GetGeometry().PointLocalCoordinates( local_point_min, min_point.Coordinates( ) ) ;
                it_cond->GetGeometry().PointLocalCoordinates( local_point_max, max_point.Coordinates( ) ) ;
                it_cond->GetGeometry().ShapeFunctionsValues( N_min, local_point_min );
                it_cond->GetGeometry().ShapeFunctionsValues( N_max, local_point_max );
            
                const array_1d<double,3>& normal_min = MortarUtilities::GaussPointUnitNormal(N_min, it_cond->GetGeometry());
                const array_1d<double,3>& normal_max = MortarUtilities::GaussPointUnitNormal(N_max, it_cond->GetGeometry());
                
                ContactUtilities::ScaleNode<NodeType>(min_point, normal_min, length_search);
                ContactUtilities::ScaleNode<NodeType>(max_point, normal_max, length_search);
                
                number_points_found = tree_points.SearchInBox(min_point, max_point, points_found.begin(), allocation_size);
            }
            else
            {
                KRATOS_ERROR << " The type search is not implemented yet does not exist!!!!. SearchTreeType = " << type_search << std::endl;
            }
            
            if (number_points_found > 0)
            {   
                // We resize the vector to the actual size
//                 points_found.resize(number_points_found); // NOTE: May be ineficient
                
            #ifdef KRATOS_DEBUG
                // NOTE: We check the list
                for (unsigned int i_point = 0; i_point < number_points_found; ++i_point )
                {
                    points_found[i_point]->Check();
                }
//                 std::cout << "The search is properly done" << std::endl;
            #endif
                
                IndexMap::Pointer indexes_map = it_cond->GetValue(INDEX_MAP);
                Condition::Pointer p_cond_slave = (*it_cond.base()); // MASTER
                
                // If not active we check if can be potentially in contact
                if (mCheckGap == true) CheckPotentialPairing(computing_contact_model_part, condition_id, p_cond_slave, points_found, number_points_found, indexes_map);
                else AddPotentialPairing(computing_contact_model_part, condition_id, p_cond_slave, points_found, number_points_found, indexes_map);
            }
        }
    }

    mrMainModelPart.RemoveConditions(TO_ERASE);
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::AddPairing(
    ModelPart& ComputingModelPart,
    std::size_t& ConditionId,
    Condition::Pointer pCondSlave,
    Condition::Pointer pCondMaster,
    IndexMap::Pointer IndexesMap
    )
{
    if (mCreateAuxiliarConditions == true) // We add the ID and we create a new auxiliar condition
    {
        IndexesMap->AddNewPair(pCondMaster->Id(), ConditionId++);
        Condition::Pointer p_auxiliar_condition = ComputingModelPart.CreateNewCondition(mConditionName, ConditionId, pCondSlave->GetGeometry(), pCondSlave->pGetProperties());
        // We set the geometrical values
        p_auxiliar_condition->SetValue(PAIRED_GEOMETRY, pCondMaster->pGetGeometry());
        p_auxiliar_condition->SetValue(NORMAL, pCondSlave->GetValue(NORMAL));
        p_auxiliar_condition->SetValue(PAIRED_NORMAL, pCondMaster->GetValue(NORMAL));
        // We activate the condition and initialize it
        p_auxiliar_condition->Set(ACTIVE, true);
        p_auxiliar_condition->Initialize();
    }
    else // We just add the ID
    {
        IndexesMap->AddNewPair(pCondMaster->Id(), 0);
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::CleanMortarConditions()
{
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        
        if ( it_cond->Has(INDEX_MAP) == true)
        {
            IndexMap::Pointer indexes_map = it_cond->GetValue(INDEX_MAP);
            if (mCheckGap == true) CheckCurrentPairing(*(it_cond.base()), indexes_map);
            else CheckAllActivePairing(*(it_cond.base()), indexes_map);;
        }
    }
    
    mrMainModelPart.RemoveConditions(TO_ERASE);
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::CheckMortarConditions()
{
    // Iterate in the conditions
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());

    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        
        if (it_cond->Has(INDEX_MAP) == true)
        {
            IndexMap::Pointer ids_destination = it_cond->GetValue(INDEX_MAP);
            if (ids_destination->size() > 0) 
            {
                std::cout << "Origin condition ID:" << it_cond->Id() << " Number of pairs: " << ids_destination->size() << std::endl;
                std::cout << ids_destination->Info();
            }
        }
    }
    
    NodesArrayType& nodes_array = mrMainModelPart.GetSubModelPart("Contact").Nodes();
    const int num_nodes = static_cast<int>(nodes_array.size());
    
    for(int i = 0; i < num_nodes; ++i) 
    {
        auto it_node = nodes_array.begin() + i;
        
        if (it_node->Is(ACTIVE) == true) std::cout << "Node: " << it_node->Id() << " is active" << std::endl;
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::InvertSearch()
{
    mInvertedSearch = !mInvertedSearch;
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
inline CheckResult TreeContactSearch<TDim, TNumNodes>::CheckCondition(
    IndexMap::Pointer IndexesMap,
    const Condition::Pointer pCond1,
    const Condition::Pointer pCond2,
    const bool InvertedSearch
    )
{
    const IndexType index_1 = pCond1->Id();
    const IndexType index_2 = pCond2->Id();
    
    if (index_1 == index_2) // Avoiding "auto self-contact"
    {
        return Fail;
    }

    // Avoid conditions oriented in the same direction
    const double tolerance = 1.0e-16;
    if (norm_2(pCond1->GetValue(NORMAL) - pCond2->GetValue(NORMAL)) < tolerance)
    {
        return Fail;
    }

    if (pCond2->Is(SLAVE) == !InvertedSearch) // Otherwise will not be necessary to check
    {
        auto& indexes_map_2 = pCond2->GetValue(INDEX_MAP);
        
        if (indexes_map_2->find(index_1) != indexes_map_2->end())
        {
            return Fail;
        }
    }
    
    // To avoid to repeat twice the same condition 
    if (IndexesMap->find(index_2) != IndexesMap->end())
    {
        return AlreadyInTheMap;
    }

    return OK;
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
inline void TreeContactSearch<TDim, TNumNodes>::AddPotentialPairing(
    ModelPart& ComputingModelPart,
    std::size_t& ConditionId,
    Condition::Pointer pCondSlave,
    PointVector& PointsFound,
    const unsigned int NumberOfPointsFound,
    IndexMap::Pointer IndexesMap
    )
{
    auto& geom = pCondSlave->GetGeometry();
    for (unsigned int i_node = 0; i_node < geom.size(); ++i_node)
    {
        geom[i_node].Set(ACTIVE, true);
    }
    
    for (unsigned int i_point = 0; i_point < NumberOfPointsFound; ++i_point )
    {
        AddPairing(ComputingModelPart, ConditionId, pCondSlave, PointsFound[i_point]->GetCondition(), IndexesMap);
    }    
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
inline void TreeContactSearch<TDim, TNumNodes>::CheckAllActivePairing(
    Condition::Pointer pCondSlave,
    IndexMap::Pointer IndexesMap
    )
{
    bool active = false;
    GeometryType& geom = pCondSlave->GetGeometry();
    for (unsigned int i_node = 0; i_node < geom.size(); ++i_node)
    {
        if (geom[i_node].Is(ACTIVE) == true) 
        {
            active = true;
            break;
        }
    }
    
    if (active == false)
    {
        for (auto it_pair = IndexesMap->begin(); it_pair != IndexesMap->end(); ++it_pair )
        {
            Condition::Pointer p_cond_master = mrMainModelPart.pGetCondition(it_pair->first);
            mrMainModelPart.pGetCondition(it_pair->second)->Set(TO_ERASE, true);
            IndexesMap->RemoveId(p_cond_master->Id());
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
inline double TreeContactSearch<TDim, TNumNodes>::Radius(GeometryType& ThisGeometry) 
{ 
    double radius = 0.0; 
    const Point& center = ThisGeometry.Center(); 
        
    for(unsigned int i_node = 0; i_node < ThisGeometry.PointsNumber(); ++i_node) 
    { 
        const array_1d<double, 3>& aux_vector = center.Coordinates() - ThisGeometry[i_node].Coordinates();
            
        const double aux_value = inner_prod(aux_vector, aux_vector); 

        if(aux_value > radius) radius = aux_value; 
    } 

    return std::sqrt(radius); 
} 

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::ResetContactOperators()
{
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        if (it_cond->Is(SLAVE) == !mInvertedSearch)
        {            
            auto& condition_pointers = it_cond->GetValue(INDEX_MAP);
            
            if (condition_pointers != nullptr)
            {
                condition_pointers->clear();
//                     condition_pointers->reserve(mAllocationSize); 
            }
        }
    }   
    
    ModelPart& computing_contact_model_part = mrMainModelPart.GetSubModelPart("ComputingContact"); 
    ConditionsArrayType& computing_conditions_array = computing_contact_model_part.Conditions();
    const int num_computing_conditions = static_cast<int>(computing_conditions_array.size());

#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_computing_conditions; ++i) 
    {
        auto it_cond = computing_conditions_array.begin() + i;
        it_cond->Set(TO_ERASE, true);
    }
    
    computing_contact_model_part.RemoveConditions(TO_ERASE);
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
void TreeContactSearch<TDim, TNumNodes>::TotalResetContactOperators()
{
    ConditionsArrayType& conditions_array = mrMainModelPart.GetSubModelPart("Contact").Conditions();
    const int num_conditions = static_cast<int>(conditions_array.size());
    
#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_conditions; ++i) 
    {
        auto it_cond = conditions_array.begin() + i;
        it_cond->Set(ACTIVE, false);
        auto& condition_pointers = it_cond->GetValue(INDEX_MAP);
        if (condition_pointers != nullptr)  condition_pointers->clear();
    }   
    
    ModelPart& computing_contact_model_part = mrMainModelPart.GetSubModelPart("ComputingContact"); 
    ConditionsArrayType& computing_conditions_array = computing_contact_model_part.Conditions();
    const int num_computing_conditions = static_cast<int>(computing_conditions_array.size());

#ifdef _OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < num_computing_conditions; ++i) 
    {
        auto it_cond = computing_conditions_array.begin() + i;
        it_cond->Set(TO_ERASE, true);
    }
    
    computing_contact_model_part.RemoveConditions(TO_ERASE);
}

/***********************************************************************************/
/***********************************************************************************/

template<unsigned int TDim, unsigned int TNumNodes>
SearchTreeType TreeContactSearch<TDim, TNumNodes>::ConvertSearchTree(const std::string& str)
{
    if(str == "InRadius") 
    {
        return KdtreeInRadius;
    }
    else if(str == "InBox") 
    {
        return KdtreeInBox;
    }
    else if (str == "KDOP")
    {
        KRATOS_ERROR << "KDOP contact search: Not yet implemented" << std::endl;
        return Kdop;
    }
    else
    {
        return KdtreeInRadius;
    }
}

/***********************************************************************************/
/***********************************************************************************/

template class TreeContactSearch<2, 2>;
template class TreeContactSearch<3, 3>;
template class TreeContactSearch<3, 4>;

}  // namespace Kratos.
