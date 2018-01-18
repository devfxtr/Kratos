// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Riccardo Rossi
//                   Vicente Mataix
//                   Klaus B. Sautter
//


// Project includes
#include "includes/define.h"
#include "custom_conditions/line_load_condition.h"
#include "utilities/math_utils.h"
#include "utilities/integration_utilities.h"

namespace Kratos
{
    //******************************* CONSTRUCTOR ****************************************
    //************************************************************************************
    
    LineLoadCondition::LineLoadCondition( IndexType NewId, GeometryType::Pointer pGeometry )
        : BaseLoadCondition( NewId, pGeometry )
    {
        //DO NOT ADD DOFS HERE!!!
    }

    //************************************************************************************
    //************************************************************************************
    
    LineLoadCondition::LineLoadCondition( IndexType NewId, GeometryType::Pointer pGeometry,  PropertiesType::Pointer pProperties )
        : BaseLoadCondition( NewId, pGeometry, pProperties )
    {
    }

    //********************************* CREATE *******************************************
    //************************************************************************************
    
    Condition::Pointer LineLoadCondition::Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties
        ) const
    {
        return boost::make_shared<LineLoadCondition>(NewId, pGeom, pProperties);
    }

    //************************************************************************************
    //************************************************************************************
    
    Condition::Pointer LineLoadCondition::Create( 
        IndexType NewId, 
        NodesArrayType const& ThisNodes,  
        PropertiesType::Pointer pProperties 
        ) const
    {
        return boost::make_shared<LineLoadCondition>( NewId, GetGeometry().Create( ThisNodes ), pProperties );
    }

    //******************************* DESTRUCTOR *****************************************
    //************************************************************************************
    
    LineLoadCondition::~LineLoadCondition()
    {
    }

    //************************************************************************************
    //************************************************************************************

    void LineLoadCondition::CalculateAll( 
        MatrixType& rLeftHandSideMatrix, 
        VectorType& rRightHandSideVector,
        ProcessInfo& rCurrentProcessInfo,
        bool CalculateStiffnessMatrixFlag,
        bool CalculateResidualVectorFlag 
        )
    {
        KRATOS_TRY;
        const unsigned int number_of_nodes = GetGeometry().size();
        const unsigned int block_size = this->GetBlockSize();
        unsigned int mat_size = number_of_nodes * block_size;

        // Resizing as needed the LHS
        if ( CalculateStiffnessMatrixFlag == true ) //calculation of the matrix is required
        {
            if ( rLeftHandSideMatrix.size1() != mat_size )
            {
                rLeftHandSideMatrix.resize( mat_size, mat_size, false );
            }

            noalias( rLeftHandSideMatrix ) = ZeroMatrix( mat_size, mat_size ); //resetting LHS
        }
        //resizing as needed the RHS
        if ( CalculateResidualVectorFlag == true ) //calculation of the matrix is required
        {
            if ( rRightHandSideVector.size( ) != mat_size )
            {
                rRightHandSideVector.resize( mat_size, false );
            }

            noalias( rRightHandSideVector ) = ZeroVector( mat_size ); //resetting RHS
        }

        // update length for each time step --> reaction forces !
        if (rCurrentProcessInfo[TIME_STEPS]!=this->mTimeStep) this->UpdateMemberLength();

        // add respective loads
        if (this->Has( LINE_LOAD )) this->AddLineLoad(rRightHandSideVector);
       
        KRATOS_CATCH( "" )
    }
   

    void LineLoadCondition::AddLineLoad(VectorType& rRightHandSideVector)
     {
        KRATOS_TRY;
        const unsigned int number_of_nodes = GetGeometry().size();
        const unsigned int dimension = GetGeometry().WorkingSpaceDimension();
        const unsigned int block_size = this->GetBlockSize();

        IntegrationMethod integration_method = IntegrationUtilities::GetIntegrationMethodForExactMassMatrixEvaluation(GetGeometry());
        const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(integration_method);
        const Matrix& Ncontainer = GetGeometry().ShapeFunctionsValues(integration_method);

        


         // Vector with a loading applied to the elemnt
        array_1d<double, 3 > line_load = ZeroVector(3);
        noalias(line_load) = this->GetValue( LINE_LOAD );

        
        for ( unsigned int point_number = 0; point_number < integration_points.size(); point_number++ )
        {                   
            const double det_j = GetGeometry().DeterminantOfJacobian( integration_points[point_number] );

            const double integration_weight = GetIntegrationWeight(integration_points, point_number, det_j); 
            
            for (unsigned int ii = 0; ii < number_of_nodes; ++ii)
            {
                unsigned int base = ii * block_size;
                
                for(unsigned int k = 0; k < dimension; ++k)
                {
                    rRightHandSideVector[base + k] += integration_weight * Ncontainer( point_number, ii ) * line_load[k];
                }
            }
        } 


        rRightHandSideVector = (rRightHandSideVector/this->GetGeometry().Length())*this->mL;

        if (this->HasRotDof()) this->CalculateAndAddWorkEquivalentNodalForcesLineLoad(line_load,rRightHandSideVector);

        KRATOS_WATCH(rRightHandSideVector);
        KRATOS_CATCH("");
     }

    void LineLoadCondition::CalculateAndAddWorkEquivalentNodalForcesLineLoad(
        const Vector& rForceInput, VectorType& rRightHandSideVector) const
    {
        KRATOS_TRY;
        const int dimension = this->GetGeometry().WorkingSpaceDimension();
        //calculate orthogonal load vector
        const Vector& r_node_1 = this->mNodeA;
        const Vector& r_node_2 = this->mNodeB;

        Vector geometric_orientation = ZeroVector(dimension);
        geometric_orientation[0] = r_node_2[0]-r_node_1[0];
        geometric_orientation[1] = r_node_2[1]-r_node_1[1];
        if (dimension == 3)
        {
            geometric_orientation[2] = r_node_2[2]-r_node_1[2];
        }

        const double vector_norm_a = MathUtils<double>::Norm(geometric_orientation);
        if (vector_norm_a != 0.00) geometric_orientation /= vector_norm_a;

        Vector line_load_dir = ZeroVector(dimension);
        for (int i = 0; i < dimension; ++i)
        {
            line_load_dir[i] = rForceInput[i];
        }

        const double vector_norm_b = MathUtils<double>::Norm(line_load_dir);
        if (vector_norm_b != 0.00) line_load_dir /= vector_norm_b;

        double cos_angle = 0.00;
        for (int i = 0; i < dimension; ++i)
        {
            cos_angle += line_load_dir[i] * geometric_orientation[i];
        }

        const double sin_angle = std::sqrt(1.00 - (cos_angle*cos_angle));
        const double norm_force_vector_orth = sin_angle * vector_norm_b;


        Vector node_b = ZeroVector(dimension);
        node_b = r_node_1 + line_load_dir;

        Vector node_c = ZeroVector(dimension);
        node_c = r_node_1 + (geometric_orientation*cos_angle);

        Vector load_orthogonal_dir = ZeroVector(dimension);
        load_orthogonal_dir = node_b - node_c;
        const double vector_norm_c = MathUtils<double>::Norm(load_orthogonal_dir);
        if(vector_norm_c != 0.00) load_orthogonal_dir /= vector_norm_c;



        // now caluclate respective work equivilent nodal moments

        const double custom_moment = (norm_force_vector_orth *
            this->mL*this->mL) / 12.00;

        Vector moment_node_a = ZeroVector(dimension);
        moment_node_a = MathUtils<double>::CrossProduct(geometric_orientation,
            load_orthogonal_dir);
        moment_node_a *= custom_moment;

        if(dimension == 3)
        {        
            for (int i = 0; i < dimension; ++i)
            {
                rRightHandSideVector[(1 * dimension) + i] += moment_node_a[i];
                rRightHandSideVector[(3 * dimension) + i] -= moment_node_a[i];
            }
        }
        else if(dimension == 2)
        {
            rRightHandSideVector[2] += moment_node_a[2];
            rRightHandSideVector[5] -= moment_node_a[2];
        }
        else KRATOS_ERROR << "the conditions only works for 2D and 3D elements";


        KRATOS_CATCH("")            
    }


    void LineLoadCondition::UpdateMemberLength()
    {
        KRATOS_TRY;
        this->mL = this->GetGeometry().Length();
        this->UpdateNodalPosition();
        this->mTimeStep++;
        KRATOS_CATCH("")
    }

    void LineLoadCondition::UpdateNodalPosition()
    {
        KRATOS_TRY;
        const int dimension = this->GetGeometry().WorkingSpaceDimension();
        const unsigned int number_of_nodes = GetGeometry().size();


        this->mNodeA = ZeroVector(dimension);
        this->mNodeA[0] = this->GetGeometry()[0].X();
        this->mNodeA[1] = this->GetGeometry()[0].Y();
        if (dimension == 3)
        {
            this->mNodeA[2] = this->GetGeometry()[0].Z();
        }

        this->mNodeB = ZeroVector(dimension);
        this->mNodeB[0] = this->GetGeometry()[number_of_nodes-1].X();
        this->mNodeB[1] = this->GetGeometry()[number_of_nodes-1].Y();
        if (dimension == 3)
        {
            this->mNodeB[2] = this->GetGeometry()[number_of_nodes-1].Z();
        }


        KRATOS_CATCH("");
    }


} // Namespace Kratos


