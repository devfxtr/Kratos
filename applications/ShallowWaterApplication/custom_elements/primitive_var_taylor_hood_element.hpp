//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Miguel Maso Sotomayor
//

#if !defined (KRATOS_PRIMITIVE_VAR_TAYLOR_HOOD_ELEMNT_H_INCLUDED)
#define KRATOS_PRIMITIVE_VAR_TAYLOR_HOOD_ELEMNT_H_INCLUDED

// System includes
#include <string>
#include <iostream>


// External includes


// Project includes
#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/element.h"
#include "geometries/geometry_data.h"


namespace Kratos
{
///@addtogroup ShallowwaterApplication
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

/// Implementation of a Taylor-Hood element for shallow water flow problems.
/** Taylor-Hood type elements use a quadratic interpolation for the velocity
  * and a linear interpolation for water height, which allows them to fulfill 
  * the inf-sup condition without introducing stabilization terms.
  */
class PrimitiveVarTaylorHoodElement : public Element
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of PrimitiveVarTaylorHoodElement
    KRATOS_CLASS_POINTER_DEFINITION(PrimitiveVarTaylorHoodElement);

    /// Type for shape function values container
    typedef Kratos::Vector              ShapeFunctionsType;

    /// Type for shape function derivatives container
    typedef Kratos::Matrix            ShapeDerivativesType;

    /// Type for geometry calls
    typedef Kratos::Geometry< Node<3> >       GeometryType;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    PrimitiveVarTaylorHoodElement(IndexType NewId = 0) :
        Element(NewId)
    {}

    /// Constructor using a Geometry instance
    PrimitiveVarTaylorHoodElement(IndexType NewId, GeometryType::Pointer pGeometry) :
        Element(NewId, pGeometry)
    {}

    /// Constructor using geometry and properties
    PrimitiveVarTaylorHoodElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties) :
        Element(NewId, pGeometry, pProperties)
    {}

    /// Destructor.
    virtual ~PrimitiveVarTaylorHoodElement()
    {}

    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    /// Create a new PrimitiveVarTaylorHoodElement element and return a pointer to it
    Element::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties)  const;

    /// Check that all required data containers are properly initialized and registered in Kratos
    /** @return 0 if no errors are detected.
      */
    virtual int Check(const ProcessInfo &rCurrentProcessInfo);

    /// Calculate Shape function derivatives and Jacobian at each integration point
    virtual void Initialize();

    /// Evaluate the elemental contribution to the problem for turbulent viscosity.
    /**
     * @param rLeftHandSideMatrix Elemental left hand side matrix
     * @param rRightHandSideVector Elemental right hand side vector
     * @param rCurrentProcessInfo Reference to the ProcessInfo from the ModelPart containg the element
     */
    virtual void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, ProcessInfo& rCurrentProcessInfo);

    virtual void CalculateRightHandSide(VectorType& rRightHandSideVector, ProcessInfo& rCurrentProcessInfo)
    {
        MatrixType TmpLHS;
        this->CalculateLocalSystem(TmpLHS,rRightHandSideVector,rCurrentProcessInfo);
    }

    /// Evaluate the elemental mass matrix, adding acceleration terms to the momentum equation
    /**
      * @param rMassMatrix Elemental mass matrix
      * @param rCurrentProcessInfo ProcessInfo instance (unused)
      */
    virtual void MassMatrix(MatrixType &rMassMatrix, ProcessInfo &rCurrentProcessInfo);

    /// Fill given array with containing the element's degrees of freedom
    virtual void GetDofList(DofsVectorType& rElementalDofList, ProcessInfo& rCurrentProcessInfo);

    /// Fill given vector with the linear system row index for the element's degrees of freedom
    virtual void EquationIdVector(Element::EquationIdVectorType& rResult, ProcessInfo& rCurrentProcessInfo);

    //~ virtual void GetValuesVector(Vector& rValues, int Step = 0);

    virtual void GetFirstDerivativesVector(Vector &rValues, int Step = 0);

    virtual void GetSecondDerivativesVector(Vector &rValues, int Step = 0);

    /// Update the pressure value for the velocity-only nodes, so it is properly printed in the postprocess.
    virtual void FinalizeSolutionStep(ProcessInfo &rCurrentProcessInfo);

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
        buffer << "PrimitiveVarTaylorHoodElement" << this->GetGeometry().WorkingSpaceDimension() << "D #" << Id();
        return buffer.str();
    }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "PrimitiveVarTaylorHoodElement" << this->GetGeometry().WorkingSpaceDimension() << "D #" << Id() << std::endl;
        rOStream << "Number of Velocity Nodes: " << this->GetGeometry().PointsNumber() << std::endl;
        rOStream << "Number of Pressure Nodes: " << mpPressureGeometry->PointsNumber() << std::endl;
        rOStream << "Integration method: " << this->mIntegrationMethod;
    }

    /// Print object's data.
    virtual void PrintData(std::ostream& rOStream) const
    {
        this->PrintInfo(rOStream);
        rOStream << "Velocity Geometry Data: " << std::endl;
        this->GetGeometry().PrintData(rOStream);
        rOStream << "Pressure  Geometry Data: " << std::endl;
        mpPressureGeometry->PrintData(rOStream);
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

    void AddMassTerm(MatrixType &rMassMatrix,
                     const ShapeFunctionsType &N,
                     const double Weight);

    void AddMomentumTerms(MatrixType &rLHS,
                          VectorType &rRHS,
                          const Vector& UGradN,
                          const double Density,
                          const double Viscosity,
                          const array_1d<double,3>& BodyForce,
                          const ShapeFunctionsType &Nv,
                          const ShapeDerivativesType &DNv_DX,
                          const double Weigth);

    void AddContinuityTerms(MatrixType &rLHS,
                            const ShapeFunctionsType &Np,
                            const ShapeDerivativesType &DNv_DX,
                            const double Weight);

    void EvaluateConvection(Vector& rResult,
                            const array_1d<double,3>& rConvVel,
                            const ShapeDerivativesType& DN_DX);

    template< class TVariableType >
    void EvaluateInPoint(TVariableType& rResult,
                         const Kratos::Variable<TVariableType> Var,
                         const ShapeFunctionsType& rShapeFunc,
                         GeometryType& rGeom)
    {
        rResult = rShapeFunc[0] * rGeom[0].FastGetSolutionStepValue(Var);

        for(SizeType i = 1; i < rShapeFunc.size(); i++)
        {
            rResult += rShapeFunc[i] * rGeom[i].FastGetSolutionStepValue(Var);
        }
    }

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

    /// Pointer to Pressure Geometry (nodes only on vertices).
    Geometry< Node<3> >::Pointer mpPressureGeometry;

    /// Order of integration.
    Element::IntegrationMethod mIntegrationMethod;

    /// Velocity shape function derivatives at each integration point.
    std::vector< ShapeDerivativesType > mDNv_DX;

    /// Pressure shape function derivatives at each integration point.
    std::vector< ShapeDerivativesType > mDNp_DX;

    /// Determinant of the Jacobian, evaluated at each integration point.
    /** Note that, for elements with straight edges, the Jacobian is constant
      * in the element.
      */
    std::vector< double > mDetJ;

    ///@}
    ///@name Serialization
    ///@{

    friend class Serializer;

    virtual void save(Serializer& rSerializer) const
    {
        KRATOS_TRY;
        KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element );
        rSerializer.save("mpPressureGeometry",mpPressureGeometry);
        unsigned int IntMethod = 0;
        switch(mIntegrationMethod)
        {
        case GeometryData::GI_GAUSS_1:
            IntMethod = 1;
            break;
        case GeometryData::GI_GAUSS_2:
            IntMethod = 2;
            break;
        case GeometryData::GI_GAUSS_3:
            IntMethod = 3;
            break;
        case GeometryData::GI_GAUSS_4:
            IntMethod = 4;
            break;
        case GeometryData::GI_GAUSS_5:
            IntMethod = 5;
            break;
        default:
            KRATOS_ERROR << "Unknown integration method encountered on serializer save for PrimitiveVarTaylorHoodElement element: " << mIntegrationMethod << std::endl;
            break;
        }
        rSerializer.save("IntMethod",IntMethod);
        rSerializer.save("mDNv_DX",mDNv_DX);
        rSerializer.save("mDNp_DX",mDNp_DX);
        rSerializer.save("mDetJ",mDetJ);
        KRATOS_CATCH("");
    }

    virtual void load(Serializer& rSerializer)
    {
        KRATOS_TRY;
        KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer,Element);
        rSerializer.load("mpPressureGeometry",mpPressureGeometry);
        unsigned int IntMethod = 0;
        rSerializer.load("IntMethod",IntMethod);
        switch(mIntegrationMethod)
        {
        case 1:
            mIntegrationMethod = GeometryData::GI_GAUSS_1;
            break;
        case 2:
            mIntegrationMethod = GeometryData::GI_GAUSS_2;
            break;
        case 3:
            mIntegrationMethod = GeometryData::GI_GAUSS_3;
            break;
        case 4:
            mIntegrationMethod = GeometryData::GI_GAUSS_4;
            break;
        case 5:
            mIntegrationMethod = GeometryData::GI_GAUSS_5;
            break;
        default:
            KRATOS_ERROR << "Unknown integration method encountered on serializer load for PrimitiveVarTaylorHoodElement element: " << IntMethod << std::endl;
            break;
        }
        rSerializer.load("mDNv_DX",mDNv_DX);
        rSerializer.load("mDNp_DX",mDNp_DX);
        rSerializer.load("mDetJ",mDetJ);
        KRATOS_CATCH("");
    }

    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{

    template < class TValueType >
    inline void ThreadSafeNodeWrite(NodeType& rNode, Variable<TValueType> Var, const TValueType Value)
    {
        rNode.SetLock();
        rNode.FastGetSolutionStepValue(Var) = Value;
        rNode.UnSetLock();
    }


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
    PrimitiveVarTaylorHoodElement& operator=(PrimitiveVarTaylorHoodElement const& rOther);

    /// Copy constructor.
    PrimitiveVarTaylorHoodElement(PrimitiveVarTaylorHoodElement const& rOther);


    ///@}

}; // Class PrimitiveVarTaylorHoodElement

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
inline std::istream& operator >> (std::istream& rIStream,
                                  PrimitiveVarTaylorHoodElement& rThis)
{
    return rIStream;
}

/// output stream function
inline std::ostream& operator << (std::ostream& rOStream,
                                  const PrimitiveVarTaylorHoodElement& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}
///@}

///@} addtogroup block

}  // namespace Kratos.

#endif // KRATOS_PRIMITIVE_VAR_TAYLOR_HOOD_ELEMNT_H_INCLUDED
