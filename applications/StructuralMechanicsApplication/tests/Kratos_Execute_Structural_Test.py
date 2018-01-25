from __future__ import print_function, absolute_import, division  # makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication

import os
import process_factory

class Kratos_Execute_Test:

    def __init__(self, project_parameters_file_name):
        self.project_parameters_file_name = project_parameters_file_name

    #### These functions should be in the base class to provide a consistent Interface
    #### This base-class should be in the core such that the applicatins can derive from it

    def Run(self):
        self.Initialize()
        self.RunMainTemporalLoop()
        self.Finalize()


    def Initialize(self):
        self.ImportAndCreateSolver()
        self.InitializeIO()
        self.ExecuteInitialize()
        self.ExecuteBeforeSolutionLoop()

    def SolveSolutionStep(self):
        self.ExecuteInitializeSolutionStep()
        self.ExecuteBeforeSolve()
        self.Solve()
        self.ExecuteAfterSolve()
        self.ExecuteFinalizeSolutionStep()
        self.ExecuteBeforeOutputStep()
        self.PrintOutput()
        self.ExecuteAfterOutputStep()

    def RunMainTemporalLoop(self):
        while time < end_time:
            self.SolveSolutionStep()
            
    def Finalize(self):
        self.ExecuteFinalize()

    ###########################################################################

    def ImportAndCreateSolver(self):
        parameter_file = open(self.project_parameters_file_name,'r')
        self.ProjectParameters = KratosMultiphysics.Parameters( parameter_file.read())

        self.ProjectParameters = ProjectParameters

        self.main_model_part = KratosMultiphysics.ModelPart(self.ProjectParameters["problem_data"]["model_part_name"].GetString())
        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, self.ProjectParameters["problem_data"]["domain_size"].GetInt())

        # Construct the solver (main setting methods are located in the solver_module)
        import python_solvers_wrapper_structural
        self.solver = python_solvers_wrapper_structural.CreateSolver(self.main_model_part, self.ProjectParameters)

        # Add variables (always before importing the model part) (it must be integrated in the ImportModelPart)
        # If we integrate it in the model part we cannot use combined solvers
        self.solver.AddVariables()

        # Read model_part (note: the buffer_size is set here) (restart can be read here)
        self.solver.ImportModelPart()

        # Add dofs (always after importing the model part) (it must be integrated in the ImportModelPart)
        # If we integrate it in the model part we cannot use combined solvers
        self.solver.AddDofs()

        self.Model = KratosMultiphysics.Model()
        self.Model.AddModelPart(self.main_model_part)

    def InitializeIO(self):
        # ### Output settings start ####
        self.output_post = ProjectParameters.Has("output_configuration")
        if (self.output_post == True):
            from gid_output_process import GiDOutputProcess
            output_settings = ProjectParameters["output_configuration"]
            self.gid_output = GiDOutputProcess(self.solver.GetComputingModelPart(),
                                               self.ProjectParameters["problem_data"]["problem_name"].GetString(),
                                               output_settings)
            self.gid_output.ExecuteInitialize()

    def ExecuteInitialize(self):
        # Obtain the list of the processes to be applied
        self.list_of_processes = process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["constraints_process_list"])
        self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["loads_process_list"])
        if (ProjectParameters.Has("list_other_processes") == True): 
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["list_other_processes"])
        if (ProjectParameters.Has("json_check_process") == True): 
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["json_check_process"]) 
        if (ProjectParameters.Has("check_analytic_results_process") == True): 
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["check_analytic_results_process"]) 
        if (ProjectParameters.Has("json_output_process") == True): 
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["json_output_process"]) 

        for process in self.list_of_processes:
            process.ExecuteInitialize()

        # ### START SOLUTION ####

        # Sets strategies, builders, linear solvers, schemes and solving info, and fills the buffer
        self.solver.Initialize()
        self.solver.SetEchoLevel(0) # Avoid to print anything 
        
    def ExecuteBeforeSolutionLoop(self):
        if (self.output_post == True):
            self.gid_output.ExecuteBeforeSolutionLoop()

        for process in self.list_of_processes:
            process.ExecuteBeforeSolutionLoop()

        # #Stepping and time settings (get from process info or solving info)
        # Delta time
        self.delta_time = self.ProjectParameters["problem_data"]["time_step"].GetDouble()
        # Start step
        self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] = 0
        # Start time
        self.time = self.ProjectParameters["problem_data"]["start_time"].GetDouble()
        # End time
        self.end_time = self.ProjectParameters["problem_data"]["end_time"].GetDouble()

    def ExecuteInitializeSolutionStep(self):
        time = time + delta_time
        self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] += 1
        self.main_model_part.CloneTimeStep(time)

        for process in self.list_of_processes:
            process.ExecuteInitializeSolutionStep()
            
        if (self.output_post == True):
            self.gid_output.ExecuteInitializeSolutionStep()

    def ExecuteBeforeSolve(self):
        # This function is not needed in this solver
        pass

    def Solve(self):
        self.solver.Solve()

    def ExecuteAfterSolve(self):
        # This function is not needed in this solver
        pass
                        
    def ExecuteFinalizeSolutionStep(self):            
        if (self.output_post == True):
            self.gid_output.ExecuteFinalizeSolutionStep()

        for process in self.list_of_processes:
            process.ExecuteFinalizeSolutionStep()

    def ExecuteBeforeOutputStep(self):
        for process in self.list_of_processes:
            process.ExecuteBeforeOutputStep()

    def PrintOutput(self):
        if (self.output_post == True):
            if self.gid_output.IsOutputStep():
                self.gid_output.PrintOutput()

    def ExecuteAfterOutputStep(self):
        for process in self.list_of_processes:
            process.ExecuteAfterOutputStep()

    def ExecuteFinalize(self):
        if (self.output_post == True):
            self.gid_output.ExecuteFinalize()

        for process in self.list_of_processes:
            process.ExecuteFinalize()
