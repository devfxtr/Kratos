from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

#import kratos core and applications
from KratosMultiphysics import *
from KratosMultiphysics.StructuralMechanicsApplication import *
from KratosMultiphysics.ExternalSolversApplication import *
from KratosMultiphysics.ShapeOptimizationApplication import *

# For time measures
import time as timer

# ======================================================================================================================================
# Model part & solver
# ======================================================================================================================================

#import define_output
parameter_file = open("ProjectParameters.json",'r')
ProjectParameters = Parameters( parameter_file.read())

#set echo level
echo_level = ProjectParameters["problem_data"]["echo_level"].GetInt()

#defining the model_part
main_model_part = ModelPart(ProjectParameters["problem_data"]["model_part_name"].GetString())
main_model_part.ProcessInfo.SetValue(DOMAIN_SIZE, ProjectParameters["problem_data"]["domain_size"].GetInt())

# Create an optimizer
# Note that internally variables related to the optimizer are added to the model part
optimizerFactory = __import__("optimizer_factory")
optimizer = optimizerFactory.CreateOptimizer( main_model_part, ProjectParameters["optimization_settings"] )

# Create solver for all response functions specified in the optimization settings
# Note that internally variables related to the individual functions are added to the model part
responseFunctionFactory = __import__("response_function_factory")
listOfResponseFunctions = responseFunctionFactory.CreateListOfResponseFunctions( main_model_part, ProjectParameters["optimization_settings"] )

# Create solver to perform structural analysis
solver_module = __import__(ProjectParameters["solver_settings"]["solver_type"].GetString())
CSM_solver = solver_module.CreateSolver(main_model_part, ProjectParameters["solver_settings"])
CSM_solver.AddVariables()
CSM_solver.ImportModelPart()

# Add degrees of freedom
CSM_solver.AddDofs()

# Create Model
model = Model()
model.AddModelPart(main_model_part)

# Build sub_model_parts or submeshes (rearrange parts for the application of custom processes)
## Get the list of the submodel part in the object Model
for i in range(ProjectParameters["solver_settings"]["processes_sub_model_part_list"].size()):
    part_name = ProjectParameters["solver_settings"]["processes_sub_model_part_list"][i].GetString()
    model.AddModelPart(main_model_part.GetSubModelPart(part_name))

# ======================================================================================================================================
# Analyzer
# ======================================================================================================================================

class kratosCSMAnalyzer( (__import__("analyzer_base")).analyzerBaseClass ):

    # --------------------------------------------------------------------------
    def initializeBeforeOptimizationLoop( self ):
        self.__initializeGIDOutput()
        self.__initializeProcesses()
        self.__initializeSolutionLoop()

    # --------------------------------------------------------------------------
    def analyzeDesignAndReportToCommunicator( self, currentDesign, optimizationIteration, communicator ):

        eigenfrequency_factor = -1.0 # maximization of eigenvalues -> negative factor

        # Calculation of value of objective function
        if communicator.isRequestingFunctionValueOf("eigenfrequency"):

            print("\n> Starting StructuralMechanicsApplication to solve structure")
            startTime = timer.time()
            self.__solveStructure( optimizationIteration )
            print("> Time needed for solving the structure = ",round(timer.time() - startTime,2),"s")

            print("\n> Starting calculation of eigenfrequency")
            startTime = timer.time()
            listOfResponseFunctions["eigenfrequency"].CalculateValue()
            print("> Time needed for calculation of eigenfrequency = ",round(timer.time() - startTime,2),"s")

            communicator.reportFunctionValue("eigenfrequency", eigenfrequency_factor * listOfResponseFunctions["eigenfrequency"].GetValue())

        # Calculation of gradient of objective function
        if communicator.isRequestingGradientOf("eigenfrequency"):

            print("\n> Starting calculation of gradients of eigenfrequency")
            startTime = timer.time()
            listOfResponseFunctions["eigenfrequency"].CalculateGradient()
            print("> Time needed for calculating gradients of eigenfrequency = ",round(timer.time() - startTime,2),"s")

            gradientForCompleteModelPart = listOfResponseFunctions["eigenfrequency"].GetGradient()
            for node_id in gradientForCompleteModelPart:
                gradient = gradientForCompleteModelPart[node_id]
                gradientForCompleteModelPart[node_id] = [eigenfrequency_factor*gradient[0], eigenfrequency_factor*gradient[1], eigenfrequency_factor*gradient[2]]
            communicator.reportGradient("eigenfrequency", gradientForCompleteModelPart)

        # Calculation of value of constraint function
        if communicator.isRequestingFunctionValueOf("mass"):

            print("\n> Starting calculation of mass")
            listOfResponseFunctions["mass"].CalculateValue()
            constraintFunctionValue = listOfResponseFunctions["mass"].GetValue() - listOfResponseFunctions["mass"].GetInitialValue()
            print("> Time needed for calculation of mass = ",round(timer.time() - startTime,2),"s")

            communicator.reportFunctionValue("mass", constraintFunctionValue)
            communicator.setFunctionReferenceValue("mass", listOfResponseFunctions["mass"].GetInitialValue())

        # Calculation of gradients of constraint function
        if communicator.isRequestingGradientOf("mass"):

            print("\n> Starting calculation of gradient of mass")
            startTime = timer.time()
            listOfResponseFunctions["mass"].CalculateGradient()
            print("> Time needed for calculating gradient of mass = ",round(timer.time() - startTime,2),"s")

            gradientForCompleteModelPart = listOfResponseFunctions["mass"].GetGradient()
            communicator.reportGradient("mass", gradientForCompleteModelPart)

    # --------------------------------------------------------------------------
    def finalizeAfterOptimizationLoop( self ):
        for process in self.list_of_processes:
            process.ExecuteFinalize()
        self.gid_output.ExecuteFinalize()

    # --------------------------------------------------------------------------
    def __initializeProcesses( self ):

        import process_factory
        #the process order of execution is important
        self.list_of_processes  = process_factory.KratosProcessFactory(model).ConstructListOfProcesses( ProjectParameters["constraints_process_list"] )
        self.list_of_processes += process_factory.KratosProcessFactory(model).ConstructListOfProcesses( ProjectParameters["loads_process_list"] )
        if (ProjectParameters.Has("list_other_processes")):
            self.list_of_processes += process_factory.KratosProcessFactory(model).ConstructListOfProcesses(ProjectParameters["list_other_processes"])
        if(ProjectParameters.Has("problem_process_list")):
            self.list_of_processes += process_factory.KratosProcessFactory(model).ConstructListOfProcesses( ProjectParameters["problem_process_list"] )
        if (ProjectParameters.Has("json_output_process")):
            self.list_of_processes += process_factory.KratosProcessFactory(model).ConstructListOfProcesses(ProjectParameters["json_output_process"])
        if(ProjectParameters.Has("output_process_list")):
            self.list_of_processes += process_factory.KratosProcessFactory(model).ConstructListOfProcesses( ProjectParameters["output_process_list"] )

        #print list of constructed processes
        if(echo_level>1):
            for process in self.list_of_processes:
                print(process)

        for process in self.list_of_processes:
            process.ExecuteInitialize()

    # --------------------------------------------------------------------------
    def __initializeGIDOutput( self ):

        computing_model_part = CSM_solver.GetComputingModelPart()
        problem_name = ProjectParameters["problem_data"]["problem_name"].GetString()

        from gid_output_process import GiDOutputProcess
        output_settings = ProjectParameters["output_configuration"]
        self.gid_output = GiDOutputProcess(computing_model_part, problem_name, output_settings)

        self.gid_output.ExecuteInitialize()

    # --------------------------------------------------------------------------
    def __initializeSolutionLoop( self ):

        ## Sets strategies, builders, linear solvers, schemes and solving info, and fills the buffer
        CSM_solver.Initialize()
        CSM_solver.SetEchoLevel(echo_level)

        for responseFunctionId in listOfResponseFunctions:
            listOfResponseFunctions[responseFunctionId].Initialize()

        # Start process
        for process in self.list_of_processes:
            process.ExecuteBeforeSolutionLoop()

        ## Set results when are written in a single file
        self.gid_output.ExecuteBeforeSolutionLoop()

    # --------------------------------------------------------------------------
    def __solveStructure( self, optimizationIteration ):

        # processes to be executed at the begining of the solution step
        for process in self.list_of_processes:
            process.ExecuteInitializeSolutionStep()

        self.gid_output.ExecuteInitializeSolutionStep()

        # Actual solution
        CSM_solver.Solve()

        for process in self.list_of_processes:
            process.ExecuteFinalizeSolutionStep()

        self.gid_output.ExecuteFinalizeSolutionStep()

        # processes to be executed at the end of the solution step
        for process in self.list_of_processes:
            process.ExecuteFinalizeSolutionStep()

        # processes to be executed before witting the output
        for process in self.list_of_processes:
            process.ExecuteBeforeOutputStep()

        # write output results GiD: (frequency writing is controlled internally)
        if(self.gid_output.IsOutputStep()):
            self.gid_output.PrintOutput()

        # processes to be executed after witting the output
        for process in self.list_of_processes:
            process.ExecuteAfterOutputStep()

    # --------------------------------------------------------------------------
    def finalizeSolutionLoop( self ):
        for process in self.list_of_processes:
            process.ExecuteFinalize()
        self.gid_output.ExecuteFinalize()

    # --------------------------------------------------------------------------

structureAnalyzer = kratosCSMAnalyzer()

# ======================================================================================================================================
# Optimization
# ======================================================================================================================================

optimizer.importAnalyzer( structureAnalyzer )
optimizer.optimize()

# ======================================================================================================================================