# ==============================================================================
#  KratosShapeOptimizationApplication
#
#  License:         BSD License
#                   license: ShapeOptimizationApplication/license.txt
#
#  Main authors:    Baumgaertner Daniel, https://github.com/dbaumgaertner
#                   Geiser Armin, https://github.com/armingeiser
#
# ==============================================================================

# Making KratosMultiphysics backward compatible with python 2.6 and 2.7
from __future__ import print_function, absolute_import, division

# importing the Kratos Library
from KratosMultiphysics import *
from KratosMultiphysics.ShapeOptimizationApplication import *

# check that KratosMultiphysics was imported in the main script
CheckForPreviousImport()

# Import algorithm base classes
from algorithm_base import OptimizationAlgorithm

# Additional imports
import timer_factory as timer_factory

# ==============================================================================
class AlgorithmPenalizedProjection( OptimizationAlgorithm ) :

    # --------------------------------------------------------------------------
    def __init__( self,
                  ModelPartController,
                  Analyzer,
                  Communicator,
                  Mapper,
                  DataLogger,
                  OptimizationSettings ):

        self.ModelPartController = ModelPartController
        self.Analyzer = Analyzer
        self.Communicator = Communicator
        self.Mapper = Mapper
        self.DataLogger = DataLogger
        self.OptimizationSettings = OptimizationSettings

        self.OptimizationModelPart = ModelPartController.GetOptimizationModelPart()
        self.DesignSurface = ModelPartController.GetDesignSurface()

        self.maxIterations = OptimizationSettings["optimization_algorithm"]["max_iterations"].GetInt() + 1
        self.projectionOnNormalsIsSpecified = OptimizationSettings["optimization_algorithm"]["project_gradients_on_surface_normals"].GetBool()
        self.onlyObjective = OptimizationSettings["objectives"][0]["identifier"].GetString()
        self.onlyConstraint = OptimizationSettings["constraints"][0]["identifier"].GetString()
        self.typeOfOnlyConstraint = OptimizationSettings["constraints"][0]["type"].GetString()
        self.dampingIsSpecified = OptimizationSettings["design_variables"]["damping"]["perform_damping"].GetBool()

        self.GeometryUtilities = GeometryUtilities( self.DesignSurface )
        self.OptimizationUtilities = OptimizationUtilities( self.DesignSurface, OptimizationSettings )
        if self.dampingIsSpecified:
            damping_regions = self.ModelPartController.GetDampingRegions()
            self.DampingUtilities = DampingUtilities( self.DesignSurface, damping_regions, self.OptimizationSettings )

    # --------------------------------------------------------------------------
    def execute( self ):
        self.__initializeOptimizationLoop()
        self.__runOptimizationLoop()
        self.__finalizeOptimizationLoop()

    # --------------------------------------------------------------------------
    def __initializeOptimizationLoop( self ):
        self.Analyzer.initializeBeforeOptimizationLoop()
        self.ModelPartController.InitializeMeshController()
        self.DataLogger.StartTimer()
        self.DataLogger.InitializeDataLogging()

    # --------------------------------------------------------------------------
    def __runOptimizationLoop( self ):

        for self.optimizationIteration in range(1,self.maxIterations):
            print("\n>===================================================================")
            print("> ",self.DataLogger.GetTimeStamp(),": Starting optimization iteration ", self.optimizationIteration)
            print(">===================================================================\n")

            self.__initializeModelPartForNewSolutionStep()

            self.__updateMeshAccordingCurrentShapeUpdate()

            self.__callCoumminicatorToCreateNewRequests()

            self.__callAnalyzerToPerformRequestedAnalyses()

            self.__storeResultOfSensitivityAnalysisOnNodes()

            if self.projectionOnNormalsIsSpecified:
                self.__projectSensitivitiesOnLocalSurfaceNormal()

            if self.dampingIsSpecified:
                self.__dampSensitivities()

            self.__computeShapeUpdate()

            if self.dampingIsSpecified:
                self.__dampShapeUpdate()

            self.__logCurrentOptimizationStep()

            self.__timeOptimizationStep()

            if self.__isAlgorithmConverged():
                break
            else:
                self.__determineAbsoluteChanges()

    # --------------------------------------------------------------------------
    def __finalizeOptimizationLoop( self ):
        self.DataLogger.FinalizeDataLogging()
        self.Analyzer.finalizeAfterOptimizationLoop()

    # --------------------------------------------------------------------------
    def __initializeModelPartForNewSolutionStep( self ):
        self.ModelPartController.CloneTimeStep( self.optimizationIteration )

    # --------------------------------------------------------------------------
    def __updateMeshAccordingCurrentShapeUpdate( self ):
        self.ModelPartController.UpdateMeshAccordingInputVariable( SHAPE_UPDATE )
        self.ModelPartController.SetReferenceMeshToMesh()

    # --------------------------------------------------------------------------
    def __callCoumminicatorToCreateNewRequests( self ):
        self.Communicator.initializeCommunication()
        self.Communicator.requestFunctionValueOf( self.onlyObjective )
        self.Communicator.requestFunctionValueOf( self.onlyConstraint )
        self.Communicator.requestGradientOf( self.onlyObjective )
        self.Communicator.requestGradientOf( self.onlyConstraint )

    # --------------------------------------------------------------------------
    def __callAnalyzerToPerformRequestedAnalyses( self ):
        self.Analyzer.analyzeDesignAndReportToCommunicator( self.DesignSurface, self.optimizationIteration, self.Communicator )
        self.__ResetPossibleShapeModificationsDuringAnalysis()

    # --------------------------------------------------------------------------
    def __ResetPossibleShapeModificationsDuringAnalysis( self ):
        self.ModelPartController.SetMeshToReferenceMesh()
        self.ModelPartController.SetDeformationVariablesToZero()

    # --------------------------------------------------------------------------
    def __storeResultOfSensitivityAnalysisOnNodes( self ):
        gradientOfObjectiveFunction = self.Communicator.getReportedGradientOf ( self.onlyObjective )
        gradientOfConstraintFunction = self.Communicator.getReportedGradientOf ( self.onlyConstraint )
        self.__storeGradientOnNodalVariable( gradientOfObjectiveFunction, OBJECTIVE_SENSITIVITY )
        self.__storeGradientOnNodalVariable( gradientOfConstraintFunction, CONSTRAINT_SENSITIVITY )

    # --------------------------------------------------------------------------
    def __storeGradientOnNodalVariable( self, gradients, variable_name ):
        for nodeId in gradients:
            gradient = Vector(3)
            gradient[0] = gradients[nodeId][0]
            gradient[1] = gradients[nodeId][1]
            gradient[2] = gradients[nodeId][2]
            self.OptimizationModelPart.Nodes[nodeId].SetSolutionStepValue(variable_name,0,gradient)

    # --------------------------------------------------------------------------
    def __projectSensitivitiesOnLocalSurfaceNormal( self ):
            self.GeometryUtilities.ComputeUnitSurfaceNormals()
            self.GeometryUtilities.ProjectNodalVariableOnUnitSurfaceNormals( OBJECTIVE_SENSITIVITY )
            self.GeometryUtilities.ProjectNodalVariableOnUnitSurfaceNormals( CONSTRAINT_SENSITIVITY )

    # --------------------------------------------------------------------------
    def __dampSensitivities( self ):
        self.DampingUtilities.DampNodalVariable( OBJECTIVE_SENSITIVITY )
        self.DampingUtilities.DampNodalVariable( CONSTRAINT_SENSITIVITY )

    # --------------------------------------------------------------------------
    def __computeShapeUpdate( self ):
        self.__mapSensitivitiesToDesignSpace()

        constraintValue = self.Communicator.getReportedFunctionValueOf( self.onlyConstraint )
        if self.__isConstraintActive( constraintValue ):
            self.OptimizationUtilities.ComputeProjectedSearchDirection()
            self.OptimizationUtilities.CorrectProjectedSearchDirection( constraintValue )
        else:
            self.OptimizationUtilities.ComputeSearchDirectionSteepestDescent()

        self.OptimizationUtilities.ComputeControlPointUpdate()
        self.__mapDesignUpdateToGeometrySpace()

    # --------------------------------------------------------------------------
    def __mapSensitivitiesToDesignSpace( self ):
        self.Mapper.MapToDesignSpace( OBJECTIVE_SENSITIVITY, MAPPED_OBJECTIVE_SENSITIVITY )
        self.Mapper.MapToDesignSpace( CONSTRAINT_SENSITIVITY, MAPPED_CONSTRAINT_SENSITIVITY )

    # --------------------------------------------------------------------------
    def __isConstraintActive( self, constraintValue ):
        if self.typeOfOnlyConstraint == "equality":
            return True
        elif constraintValue > 0:
            return True
        else:
            return False

    # --------------------------------------------------------------------------
    def __mapDesignUpdateToGeometrySpace( self ):
        self.Mapper.MapToGeometrySpace( CONTROL_POINT_UPDATE, SHAPE_UPDATE )

    # --------------------------------------------------------------------------
    def __dampShapeUpdate( self ):
        self.DampingUtilities.DampNodalVariable( SHAPE_UPDATE )

    # --------------------------------------------------------------------------
    def __logCurrentOptimizationStep( self ):
        self.DataLogger.LogCurrentData( self.optimizationIteration )

    # --------------------------------------------------------------------------
    def __timeOptimizationStep( self ):
        print("\n> Time needed for current optimization step = ", self.DataLogger.GetLapTime(), "s")
        print("> Time needed for total optimization so far = ", self.DataLogger.GetTotalTime(), "s")

    # --------------------------------------------------------------------------
    def __isAlgorithmConverged( self ):

        if self.optimizationIteration > 1 :

            # Check if maximum iterations were reached
            if self.optimizationIteration == self.maxIterations:
                print("\n> Maximal iterations of optimization problem reached!")
                return True

            relativeChangeOfObjectiveValue = self.DataLogger.GetValue( "RELATIVE_CHANGE_OF_OBJECTIVE_VALUE" )

            # Check for relative tolerance
            relativeTolerance = self.OptimizationSettings["optimization_algorithm"]["relative_tolerance"].GetDouble()
            if abs(relativeChangeOfObjectiveValue) < relativeTolerance:
                print("\n> Optimization problem converged within a relative objective tolerance of ",relativeTolerance,"%.")
                return True

            # Check if value of objective increases
            if relativeChangeOfObjectiveValue > 0:
                print("\n> Value of objective function increased!")
                return False

    # --------------------------------------------------------------------------
    def __determineAbsoluteChanges( self ):
        self.OptimizationUtilities.AddFirstVariableToSecondVariable( CONTROL_POINT_UPDATE, CONTROL_POINT_CHANGE )
        self.OptimizationUtilities.AddFirstVariableToSecondVariable( SHAPE_UPDATE, SHAPE_CHANGE )

# ==============================================================================
