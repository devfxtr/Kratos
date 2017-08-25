// KRATOS  __  __ _____ ____  _   _ ___ _   _  ____ 
//        |  \/  | ____/ ___|| | | |_ _| \ | |/ ___|
//        | |\/| |  _| \___ \| |_| || ||  \| | |  _ 
//        | |  | | |___ ___) |  _  || || |\  | |_| |
//        |_|  |_|_____|____/|_| |_|___|_| \_|\____| APPLICATION
//
//  License:		 BSD License
//                       license: MeshingApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrándiz
//

#if !defined(KRATOS_SPR_ERROR_METRICS_PROCESS)
#define KRATOS_SPR_ERROR_METRICS_PROCESS

// Project includes
#include "utilities/math_utils.h"
#include "custom_utilities/metrics_math_utils.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "meshing_application.h"
#include "processes/compute_nodal_gradient_process.h" // TODO: Not prism or quadrilaterals implemented yet
#include "processes/find_nodal_neighbours_process.h"


namespace Kratos
{
///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

    typedef ModelPart::NodesContainerType                                     NodesArrayType;
    typedef ModelPart::ElementsContainerType                               ElementsArrayType;
    typedef ModelPart::ConditionsContainerType                           ConditionsArrayType;
    typedef Node <3>                                                                NodeType;
    
///@}
///@name  Enum's
///@{
    
    #if !defined(INTERPOLATION_METRIC)
    #define INTERPOLATION_METRIC
        enum Interpolation {Constant = 0, Linear = 1, Exponential = 2};
    #endif
    
///@}
///@name  Functions
///@{
    
///@}
///@name Kratos Classes
///@{

//// This class is can be used to compute the metrics of the model part with an Hessian approach

template<unsigned int TDim, class TVarType>  
class ComputeSPRErrorSolMetricProcess
    : public Process
{
public:

    ///@name Type Definitions
    ///@{
    
    /// Pointer definition of ComputeSPRErrorSolMetricProcess
    KRATOS_CLASS_POINTER_DEFINITION(ComputeSPRErrorSolMetricProcess);
    
    ///@}
    ///@name Life Cycle
    ///@{
     
    // Constructor
    
    /**
     * This is the default constructor
     * @param rThisModelPart: The model part to be computed
     * @param ThisParameters: The input parameters
     */
    
    ComputeSPRErrorSolMetricProcess(
        ModelPart& rThisModelPart,
        TVarType& rVariable,
        Parameters ThisParameters = Parameters(R"({})")
        )
        :mThisModelPart(rThisModelPart),
        mVariable(rVariable)
    {               
        Parameters DefaultParameters = Parameters(R"(
        {
            "minimal_size"                        : 0.1,
            "maximal_size"                        : 10.0, 
            "enforce_current"                     : true, 
            "hessian_strategy_parameters": 
            { 
                "interpolation_error"                  : 1.0e-6, 
                "mesh_dependent_constant"              : 0.28125
            }, 
            "anisotropy_remeshing"                : true, 
            "anisotropy_parameters":
            {
                "hmin_over_hmax_anisotropic_ratio"     : 1.0, 
                "boundary_layer_max_distance"          : 1.0, 
                "interpolation"                        : "Linear"
            }
        })" );
        ThisParameters.ValidateAndAssignDefaults(DefaultParameters);
         
        mMinSize = ThisParameters["minimal_size"].GetDouble();
        mMaxSize = ThisParameters["maximal_size"].GetDouble();
        mEnforceCurrent = ThisParameters["enforce_current"].GetBool();
        
        // In case we have isotropic remeshing (default values)
        if (ThisParameters["anisotropy_remeshing"].GetBool() == false)
        {
            mInterpError = DefaultParameters["hessian_strategy_parameters"]["interpolation_error"].GetDouble();
            mMeshConstant = DefaultParameters["hessian_strategy_parameters"]["mesh_dependent_constant"].GetDouble();
            mAnisRatio = DefaultParameters["anisotropy_parameters"]["hmin_over_hmax_anisotropic_ratio"].GetDouble();
            mBoundLayer = DefaultParameters["anisotropy_parameters"]["boundary_layer_max_distance"].GetDouble();
            mInterpolation = ConvertInter(DefaultParameters["anisotropy_parameters"]["interpolation"].GetString());
        }
        else
        {
            mInterpError = ThisParameters["hessian_strategy_parameters"]["interpolation_error"].GetDouble();
            mMeshConstant = ThisParameters["hessian_strategy_parameters"]["mesh_dependent_constant"].GetDouble();
            mAnisRatio = ThisParameters["anisotropy_parameters"]["hmin_over_hmax_anisotropic_ratio"].GetDouble();
            mBoundLayer = ThisParameters["anisotropy_parameters"]["boundary_layer_max_distance"].GetDouble();
            mInterpolation = ConvertInter(ThisParameters["anisotropy_parameters"]["interpolation"].GetString());
        }
    }
    
    /// Destructor.
    virtual ~ComputeSPRErrorSolMetricProcess() {}
    
    ///@}
    ///@name Operators
    ///@{

    void operator()()
    {
        Execute();
    }

    ///@}
    ///@name Operations
    ///@{
    
    /**
     * We initialize the metrics of the MMG sol using the Hessian metric matrix approach
     */
    
    virtual void Execute()
    {
        // Iterate in the nodes
        //NodesArrayType& NodesArray = mThisModelPart.Nodes();
        //int numNodes = NodesArray.end() - NodesArray.begin();
        
        //CalculateAuxiliarHessian();
        CalculateSuperconvergentPatchRecovery();
        /*
        #pragma omp parallel for 
        for(int i = 0; i < numNodes; i++) 
        {
            auto itNode = NodesArray.begin() + i;
            
            if ( itNode->SolutionStepsDataHas( mVariable ) == false )
            {
                KRATOS_ERROR << "Missing variable on node " << itNode->Id() << std::endl;
            }
            
            const double Distance = itNode->FastGetSolutionStepValue(DISTANCE); // TODO: This should be changed for the varaible of interestin the future. This means that the value of the boundary value would be changed to a threshold value instead
            const Vector& Hessian = itNode->GetValue(AUXILIAR_HESSIAN);

            const double NodalH = itNode->FastGetSolutionStepValue(NODAL_H);            
            
            double ElementMinSize = mMinSize;
            if ((ElementMinSize > NodalH) && (mEnforceCurrent == true))
            {
                ElementMinSize = NodalH;
            }
            double ElementMaxSize = mMaxSize;
            if ((ElementMaxSize > NodalH) && (mEnforceCurrent == true))
            {
                ElementMaxSize = NodalH;
            }
            
            const double Ratio = CalculateAnisotropicRatio(Distance, mAnisRatio, mBoundLayer, mInterpolation);
            
            // For postprocess pourposes
            itNode->SetValue(ANISOTROPIC_RATIO, Ratio); 
            
            // We compute the metric
            #ifdef KRATOS_DEBUG 
            if( itNode->Has(MMG_METRIC) == false) 
            {
                KRATOS_ERROR <<  " MMG_METRIC not defined for node " << itNode->Id();
            }
            #endif     
            Vector& Metric = itNode->GetValue(MMG_METRIC);
            
            #ifdef KRATOS_DEBUG 
            if(Metric.size() != TDim * 3 - 3) 
            {
                KRATOS_ERROR << "Wrong size of vector MMG_METRIC found for node " << itNode->Id() << " size is " << Metric.size() << " expected size was " << TDim * 3 - 3;
            }
            #endif
            
            const double NormMetric = norm_2(Metric);
            if (NormMetric > 0.0) // NOTE: This means we combine differents metrics, at the same time means that the metric should be reseted each time
            {
                const Vector OldMetric = itNode->GetValue(MMG_METRIC);
                const Vector NewMetric = ComputeHessianMetricTensor(Hessian, Ratio, ElementMinSize, ElementMaxSize);    
                
                Metric = MetricsMathUtils<TDim>::IntersectMetrics(OldMetric, NewMetric);
            }
            else
            {
                Metric = ComputeHessianMetricTensor(Hessian, Ratio, ElementMinSize, ElementMaxSize);    
            }
            
        }*/
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
        return "ComputeSPRErrorSolMetricProcess";
    }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "ComputeSPRErrorSolMetricProcess";
    }

    /// Print object"s data.
    virtual void PrintData(std::ostream& rOStream) const
    {
    }
    
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
    ///@name Private static Member Variables
    ///@{

    ///@}
    ///@name Private member Variables
    ///@{
    
    ModelPart& mThisModelPart;               // The model part to compute
    TVarType mVariable;            // The variable to calculate the hessian
    double mMinSize;                         // The minimal size of the elements
    double mMaxSize;                         // The maximal size of the elements
    bool mEnforceCurrent;                    // With this we choose if we inforce the current nodal size (NODAL_H)
    double mInterpError;                     // The error of interpolation allowed
    double mMeshConstant;                    // The mesh constant to remesh (depends of the element type)
    double mAnisRatio;                       // The minimal anisotropic ratio (0 < ratio < 1)
    double mBoundLayer;                      // The boundary layer limit distance
    Interpolation mInterpolation;            // The interpolation type
    
    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * This function is used to compute the Hessian Metric tensor, note that when using the Hessian, more than one Metric can be defined simultaneously, so in consecuence we need to define the elipsoid which defines the volume of maximal intersection
     * @param Hessian: The hessian tensor condensed already computed
     * @param AnisotropicRatio: The anisotropic ratio
     * @param ElementMinSize: The min size of element
     * @param ElementMaxSize: The maximal size of the elements
     */
        
    Vector ComputeHessianMetricTensor(
        const Vector& Hessian,
        const double& AnisotropicRatio,
        const double& ElementMinSize, // This way we can impose as minimum as the previous size if we desire
        const double& ElementMaxSize // This way we can impose as maximum as the previous size if we desire
        )
    {        
        // Calculating Metric parameters
        const double CEpsilon = mMeshConstant/mInterpError;
        const double MinRatio = 1.0/(ElementMinSize * ElementMinSize);
//         const double MinRatio = 1.0/(mMinSize * mMinSize);
        const double MaxRatio = 1.0/(ElementMaxSize * ElementMaxSize);
//         const double MaxRatio = 1.0/(mMaxSize * mMaxSize);
        
        typedef bounded_matrix<double, TDim, TDim> TempType;
        
        // Declaring the eigen system
        bounded_matrix<double, TDim, TDim> EigenVectorMatrix;
        bounded_matrix<double, TDim, TDim> EigenValuesMatrix;

        // We first transform into a matrix
        const bounded_matrix<double, TDim, TDim> HessianMatrix = MetricsMathUtils<TDim>::VectorToTensor(Hessian);
        
        MathUtils<double>::EigenSystem<TDim>(HessianMatrix, EigenVectorMatrix, EigenValuesMatrix, 1e-18, 20);
        
        // Recalculate the Metric eigen values
        for (unsigned int i = 0; i < TDim; i++)
        {
            EigenValuesMatrix(i, i) = MathUtils<double>::Min(MathUtils<double>::Max(CEpsilon * std::abs(EigenValuesMatrix(i, i)), MaxRatio), MinRatio);
        }
        
        // Considering anisotropic
        if (AnisotropicRatio < 1.0)
        {
            double EigenMax = EigenValuesMatrix(0, 0);
            double EigenMin = EigenValuesMatrix(1, 1);
            for (unsigned int i = 1; i < TDim - 1; i++)
            {
                EigenMax = MathUtils<double>::Max(EigenMax, EigenValuesMatrix(i, i));
                EigenMin = MathUtils<double>::Min(EigenMax, EigenValuesMatrix(i, i));
            }
            
            const double EigenRadius = std::abs(EigenMax - EigenMin) * (1.0 - AnisotropicRatio);
            const double RelativeEigenRadius = std::abs(EigenMax - EigenRadius);
            
            for (unsigned int i = 0; i < TDim; i++)
            {
                EigenValuesMatrix(i, i) = MathUtils<double>::Max(MathUtils<double>::Min(EigenValuesMatrix(i, i), EigenMax), RelativeEigenRadius);
            }
        }
        else // NOTE: For isotropic we should consider the maximum of the eigenvalues
        {
            double EigenMax = EigenValuesMatrix(0, 0);
            for (unsigned int i = 1; i < TDim - 1; i++)
            {
                EigenMax = MathUtils<double>::Max(EigenMax, EigenValuesMatrix(i, i));
            }
            for (unsigned int i = 0; i < TDim; i++)
            {
                EigenValuesMatrix(i, i) = EigenMax;
            }
            EigenVectorMatrix = IdentityMatrix(TDim, TDim);
        }
            
        // We compute the product
        const bounded_matrix<double, TDim, TDim> MetricMatrix =  prod(trans(EigenVectorMatrix), prod<TempType>(EigenValuesMatrix, EigenVectorMatrix));
        
        // Finally we transform to a vector
        const Vector Metric = MetricsMathUtils<TDim>::TensorToVector(MetricMatrix);
        
        return Metric;
    }



    void CalculateSuperconvergentPatchRecovery()
    {
        /************************************************************************
        --1-- calculate superconvergent stresses (at the nodes) --1--
        ************************************************************************/
        
        //std::vector<std::string> submodels;
        //submodels= mThisModelPart.GetSubModelPartNames();
        //for (std::vector<std::string>::const_iterator i = submodels.begin();i!=submodels.end();i++) 
        //    std::cout << *i<<std::endl; 
        FindNodalNeighboursProcess findNeighbours(mThisModelPart);
        findNeighbours.Execute();
        //std::vector<Vector> stress_vector(1);
        //std::vector<array_1d<double,3>> coordinates_vector(1);
        //ariable<array_1d<double,3>> variable_coordinates = INTEGRATION_COORDINATES;
        //iteration over all nodes -- construction of patches
        ModelPart::NodesContainerType& rNodes = mThisModelPart.Nodes();
        for(ModelPart::NodesContainerType::iterator i_nodes = rNodes.begin(); i_nodes!=rNodes.end(); i_nodes++){
            int neighbour_size = i_nodes->GetValue(NEIGHBOUR_ELEMENTS).size();
            std::cout << "Node: " << i_nodes->Id() << " has " << neighbour_size << " neighbouring elements: " << std::endl;
            Vector sigma_recovered(3,0);
            if(neighbour_size>2){ 
                CalculatePatch(i_nodes,i_nodes,neighbour_size,sigma_recovered);
                i_nodes->SetValue(RECOVERED_STRESS,sigma_recovered);
                std::cout<<"recovered sigma"<<sigma_recovered<<std::endl;
            }
            else{
                for(WeakPointerVector< Node<3> >::iterator i_neighbour_nodes = i_nodes->GetValue(NEIGHBOUR_NODES).begin(); i_neighbour_nodes != i_nodes->GetValue(NEIGHBOUR_NODES).end(); i_neighbour_nodes++){
                    Vector sigma_recovered_i(3);
                    unsigned int count_i=0;
                    for(ModelPart::NodesContainerType::iterator i = rNodes.begin(); i!=rNodes.end(); i++){
                        if (i->Id() == i_neighbour_nodes->Id() && i->GetValue(NEIGHBOUR_ELEMENTS).size()>2){
                            CalculatePatch(i_nodes,i,neighbour_size,sigma_recovered_i);
                            count_i ++;
                        }
                    }
                    //average solution from different patches
                    if(count_i != 0)
                        sigma_recovered =sigma_recovered*(count_i-1)/count_i + sigma_recovered_i/count_i;
                }
                i_nodes->SetValue(RECOVERED_STRESS,sigma_recovered);
            }
       }
        /******************************************************************************
        --2-- calculate error estimation and new element size (for each element) --2--
        ******************************************************************************/
        //loop over all elements: 
        double error_overall_squared=0;
        double energy_norm_overall_squared=0;

        //compute the error estimate per element
        for(ModelPart::ElementsContainerType::iterator i_elements = mThisModelPart.Elements().begin() ; i_elements != mThisModelPart.Elements().end(); i_elements++) 
        {
            std::vector<double> error_integration_point;
            i_elements->GetValueOnIntegrationPoints(ERROR_INTEGRATION_POINT,error_integration_point,mThisModelPart.GetProcessInfo());
            double error_energy_norm=0;
            for(unsigned int i=0;i<error_integration_point.size();i++)
                error_energy_norm += error_integration_point[i];
            error_overall_squared += error_energy_norm;
            error_energy_norm= sqrt(error_energy_norm);
            i_elements->SetValue(ELEMENT_ERROR,error_energy_norm);
            std::cout<<"element_error:"<<error_energy_norm<<std::endl;


            std::vector<double> strain_energy;
            i_elements->GetValueOnIntegrationPoints(STRAIN_ENERGY,strain_energy,mThisModelPart.GetProcessInfo());
            double energy_norm=0;
            for(unsigned int i=0;i<strain_energy.size();i++)
                energy_norm += 2*strain_energy[i];
            energy_norm_overall_squared += energy_norm;
            energy_norm= sqrt(energy_norm);
            std::cout<<"energy norm:"<<energy_norm<<std::endl;
        }
        std::cout<<"overall error norm (squared):"<<error_overall_squared<<std::endl;
        std::cout<<"overall energy norm (squarde):"<<energy_norm_overall_squared<<std::endl;
        
        //compute new element size
        for(ModelPart::ElementsContainerType::iterator i_elements = mThisModelPart.Elements().begin() ; i_elements != mThisModelPart.Elements().end(); i_elements++) 
        {
            //compute the current element size h
            i_elements->CalculateElementSize();

            //compute new element size
            double new_element_size;
            new_element_size = i_elements->GetValue(ELEMENT_H)/i_elements->GetValue(ELEMENT_ERROR);
            new_element_size *= sqrt((energy_norm_overall_squared+error_overall_squared)/mThisModelPart.Elements().size())*0.05;
            std::cout<<"old element size: "<<i_elements->GetValue(ELEMENT_H)<<std::endl;
            i_elements->SetValue(ELEMENT_H,new_element_size);
            std::cout<<"new element size: "<<i_elements->GetValue(ELEMENT_H)<<std::endl;
        }

        /******************************************************************************
        --3-- calculate metric (for each node) --3--
        ******************************************************************************/

        for(ModelPart::NodesContainerType::iterator i_nodes = rNodes.begin(); i_nodes!=rNodes.end(); i_nodes++){
            // get maximal element size from neighboring elements
            double h_min=0;
            for(WeakPointerVector< Element >::iterator i_neighbour_elements = i_nodes->GetValue(NEIGHBOUR_ELEMENTS).begin(); i_neighbour_elements != i_nodes->GetValue(NEIGHBOUR_ELEMENTS).end(); i_neighbour_elements++){
                if(h_min==0||h_min>i_neighbour_elements->GetValue(ELEMENT_H))
                    h_min = i_neighbour_elements->GetValue(ELEMENT_H);
                
            }
            //std::cout<<"h_min: "<<h_min<<std::endl;
            // set metric
            Matrix metric_matrix(2,2,0);
            metric_matrix(0,0)=1/(h_min*h_min);
            metric_matrix(1,1)=1/(h_min*h_min);
            // transform metric matrix to a vector
            const Vector metric = MetricsMathUtils<TDim>::TensorToVector(metric_matrix);
            i_nodes->SetValue(MMG_METRIC,metric);


            std::cout<<"metric: "<<i_nodes->GetValue(MMG_METRIC)<<std::endl;
        }
    }
    //calculates the recovered stress at a node 
    // i_node: the node for which the recovered stress should be calculated
    // i_patch_node: the center node of the patch
    void CalculatePatch(
        ModelPart::NodesContainerType::iterator i_nodes,
        ModelPart::NodesContainerType::iterator i_patch_node,
        int neighbour_size,
        Vector& rsigma_recovered)
    {
        std::vector<Vector> stress_vector(1);
        std::vector<array_1d<double,3>> coordinates_vector(1);
        Variable<array_1d<double,3>> variable_coordinates = INTEGRATION_COORDINATES;
        Matrix A(3,3,0);
        Matrix b(3,3,0); 
        Matrix p_k(1,3,0);
        for( WeakPointerVector< Element >::iterator i_elements = i_patch_node->GetValue(NEIGHBOUR_ELEMENTS).begin(); i_elements != i_patch_node->GetValue(NEIGHBOUR_ELEMENTS).end(); i_elements++) {
            std::cout << "\tElement: " << i_elements->Id() << std::endl;
            i_elements->GetValueOnIntegrationPoints(mVariable,stress_vector,mThisModelPart.GetProcessInfo());
            i_elements->GetValueOnIntegrationPoints(variable_coordinates,coordinates_vector,mThisModelPart.GetProcessInfo());

            std::cout << "\tstress: " << stress_vector[0] << std::endl;
            std::cout << "\tx: " << coordinates_vector[0][0] << "\ty: " << coordinates_vector[0][1] << "\tz_coordinate: " << coordinates_vector[0][2] << std::endl;
            Matrix sigma(1,3);
            for(int j=0;j<3;j++)
                sigma(0,j)=stress_vector[0][j];
            p_k(0,0)=1;
            p_k(0,1)=coordinates_vector[0][0]-i_patch_node->X(); 
            p_k(0,2)=coordinates_vector[0][1]-i_patch_node->Y();   
            A+=prod(trans(p_k),p_k);
            b+=prod(trans(p_k),sigma);
        }
        Matrix invA(3,3);
        double det;
        MathUtils<double>::InvertMatrix(A,invA,det);
        //std::cout <<A<<std::endl;
        //std::cout <<invA<<std::endl;
        //std::cout << det<< std::endl;

        Matrix coeff(3,3);
        coeff = prod(invA,b);
        if(neighbour_size > 2)
            rsigma_recovered = MatrixRow(coeff,0);
        else{
            p_k(0,1)=i_nodes->X()-i_patch_node->X(); 
            p_k(0,2)=i_nodes->Y()-i_patch_node->Y();
            Matrix sigma(1,3);
            sigma = prod(p_k,coeff);
            rsigma_recovered = MatrixRow(sigma,0);
        }
    }
    
    /**
     * This calculates the auxiliar hessian needed for the Metric
     * @param rThisModelPart: The original model part where we compute the hessian
     * @param rVariable: The variable to calculate the hessian
     */
    
    void CalculateAuxiliarHessian()
    {
        // Iterate in the nodes
        NodesArrayType& NodesArray = mThisModelPart.Nodes();
        int numNodes = NodesArray.end() - NodesArray.begin();
        
        // Declaring auxiliar vector
        const Vector AuxZeroVector = ZeroVector(3 * (TDim - 1));
        
        #pragma omp parallel for
        for(int i = 0; i < numNodes; i++) 
        {
            auto itNode = NodesArray.begin() + i;
            
            itNode->SetValue(AUXILIAR_HESSIAN, AuxZeroVector);  
        }
        
        // Compute auxiliar gradient
        ComputeNodalGradientProcess<TDim, TVarType, NonHistorical> GradientProcess = ComputeNodalGradientProcess<TDim, TVarType, NonHistorical>(mThisModelPart, mVariable, AUXILIAR_GRADIENT, NODAL_AREA);
        GradientProcess.Execute();
        
        // Iterate in the conditions
        ElementsArrayType& ElementsArray = mThisModelPart.Elements();
        int numElements = ElementsArray.end() - ElementsArray.begin();
        
        #pragma omp parallel for
        for(int i = 0; i < numElements; i++) 
        {
            auto itElem = ElementsArray.begin() + i;
            
            Element::GeometryType& geom = itElem->GetGeometry();

            double Volume;
            if (geom.GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Triangle2D3)
            {
                bounded_matrix<double,3, 2> DN_DX;
                array_1d<double, 3> N;
    
                GeometryUtils::CalculateGeometryData(geom, DN_DX, N, Volume);
                
                bounded_matrix<double,3, 2> values;
                for(unsigned int iNode = 0; iNode < 3; iNode++)
                {
                    const array_1d<double, 3> AuxGrad = geom[iNode].GetValue(AUXILIAR_GRADIENT);
                    values(iNode, 0) = AuxGrad[0];
                    values(iNode, 1) = AuxGrad[1];
                }
                
                const bounded_matrix<double,2, 2> Hessian = prod(trans(DN_DX), values); 
                const Vector HessianCond = MetricsMathUtils<2>::TensorToVector(Hessian);
                
                for(unsigned int iNode = 0; iNode < geom.size(); iNode++)
                {
                    for(unsigned int k = 0; k < 3; k++)
                    {
                        double& val = geom[iNode].GetValue(AUXILIAR_HESSIAN)[k];
                        
                        #pragma omp atomic
                        val += N[iNode] * Volume * HessianCond[k];
                    }
                }
            }
            else if (geom.GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Tetrahedra3D4)
            {
                bounded_matrix<double,4,  3> DN_DX;
                array_1d<double, 4> N;
                
                GeometryUtils::CalculateGeometryData(geom, DN_DX, N, Volume);
                
                bounded_matrix<double,4, 3> values;
                for(unsigned int iNode = 0; iNode < 4; iNode++)
                {
                    const array_1d<double, 3> AuxGrad = geom[iNode].GetValue(AUXILIAR_GRADIENT);
                    values(iNode, 0) = AuxGrad[0];
                    values(iNode, 1) = AuxGrad[1];
                    values(iNode, 2) = AuxGrad[2];
                }
                
                const bounded_matrix<double, 3, 3> Hessian = prod(trans(DN_DX), values); 
                const Vector HessianCond = MetricsMathUtils<3>::TensorToVector(Hessian);
                
                for(unsigned int iNode = 0; iNode < geom.size(); iNode++)
                {
                    for(unsigned int k = 0; k < 6; k++)
                    {
                        double& val = geom[iNode].GetValue(AUXILIAR_HESSIAN)[k];
                        
                        #pragma omp atomic
                        val += N[iNode] * Volume * HessianCond[k];
                    }
                }
            }
            else
            {
                KRATOS_ERROR << "WARNING: YOU CAN USE JUST 2D TRIANGLES OR 3D TETRAEDRA RIGHT NOW IN THE GEOMETRY UTILS: " << geom.size() << std::endl;
            }
        }
            
        #pragma omp parallel for
        for(int i = 0; i < numNodes; i++) 
        {
            auto itNode = NodesArray.begin() + i;
            itNode->GetValue(AUXILIAR_HESSIAN) /= itNode->FastGetSolutionStepValue(NODAL_AREA);
        }
    }
    
    /**
     * This converts the interpolation string to an enum
     * @param str: The string that you want to comvert in the equivalent enum
     * @return Interpolation: The equivalent enum (this requires less memmory than a std::string)
     */
        
    Interpolation ConvertInter(const std::string& str)
    {
        if(str == "Constant") 
        {
            return Constant;
        }
        else if(str == "Linear") 
        {
            return Linear;
        }
        else if(str == "Exponential") 
        {
            return Exponential;
        }
        else
        {
            return Linear;
        }
    }
        
    /**
     * This calculates the anisotropic ratio
     * @param distance: Distance parameter
     */
    
    double CalculateAnisotropicRatio(
        const double& distance,
        const double& rAnisRatio,
        const double& rBoundLayer,
        const Interpolation& rInterpolation
        )
    {
        const double tolerance = 1.0e-12;
        double ratio = 1.0; // NOTE: Isotropic mesh
        if (rAnisRatio < 1.0)
        {                           
            if (std::abs(distance) <= rBoundLayer)
            {
                if (rInterpolation == Constant)
                {
                    ratio = rAnisRatio;
                }
                else if (rInterpolation == Linear)
                {
                    ratio = rAnisRatio + (std::abs(distance)/rBoundLayer) * (1.0 - rAnisRatio);
                }
                else if (rInterpolation == Exponential)
                {
                    ratio = - std::log(std::abs(distance)/rBoundLayer) * rAnisRatio + tolerance;
                    if (ratio > 1.0)
                    {
                        ratio = 1.0;
                    }
                }
            }
        }
        
        return ratio;
    }
    
    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Private LifeCycle
    ///@{
    
    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    ComputeSPRErrorSolMetricProcess& operator=(ComputeSPRErrorSolMetricProcess const& rOther);

    /// Copy constructor.
    //ComputeSPRErrorSolMetricProcess(ComputeSPRErrorSolMetricProcess const& rOther);

    ///@}
};// class ComputeSPRErrorSolMetricProcess
///@}


///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

/// input stream function
template<unsigned int TDim, class TVarType> 
inline std::istream& operator >> (std::istream& rIStream,
                                  ComputeSPRErrorSolMetricProcess<TDim, TVarType>& rThis);

/// output stream function
template<unsigned int TDim, class TVarType> 
inline std::ostream& operator << (std::ostream& rOStream,
                                  const ComputeSPRErrorSolMetricProcess<TDim, TVarType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

};// namespace Kratos.
#endif /* KRATOS_SPR_ERROR_METRICS_PROCESS defined */