from __future__ import print_function, absolute_import, division  # makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import KratosMultiphysics
import KratosMultiphysics.ExternalSolversApplication as ExternalSolversApplication
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ContactStructuralMechanicsApplication as ContactStructuralMechanicsApplication

import os
import process_factory

class Kratos_Execute_Test:

    def __init__(self, ProjectParameters, frictionless_by_components):

        self.ProjectParameters = ProjectParameters

        self.echo_level = self.ProjectParameters["problem_data"]["echo_level"].GetInt()
        self.parallel_type = self.ProjectParameters["problem_data"]["parallel_type"].GetString()

        if (frictionless_by_components == True):
            if (self.ProjectParameters.Has("output_configuration") == True):
                list_nodal_var = self.ProjectParameters["output_configuration"]["result_file_configuration"]["nodal_results"]
                for i in range(0, list_nodal_var.size()):
                    if (list_nodal_var[i].GetString() == "NORMAL_CONTACT_STRESS"):
                        list_nodal_var[i].SetString("VECTOR_LAGRANGE_MULTIPLIER")
                new_list = list_nodal_var.Clone()
                self.ProjectParameters["output_configuration"]["result_file_configuration"].RemoveValue("nodal_results")
                self.ProjectParameters["output_configuration"]["result_file_configuration"].AddValue("nodal_results", new_list)
            self.ProjectParameters["solver_settings"]["contact_settings"]["mortar_type"].SetString("ALMContactFrictionlessComponents")
            for i in range(self.ProjectParameters["contact_process_list"].size()):
                self.ProjectParameters["contact_process_list"][i]["Parameters"]["contact_type"].SetString("FrictionlessComponents")

        # To avoid many prints
        if (self.echo_level == 0):
            KratosMultiphysics.Logger.GetDefaultOutput().SetSeverity(KratosMultiphysics.Logger.Severity.WARNING)

        ## Import parallel modules if needed
        if (self.parallel_type == "MPI"):
            import KratosMultiphysics.mpi as mpi
            import KratosMultiphysics.MetisApplication as MetisApplication
            import KratosMultiphysics.TrilinosApplication as TrilinosApplication

        self.main_model_part = KratosMultiphysics.ModelPart(self.ProjectParameters["problem_data"]["model_part_name"].GetString())
        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, self.ProjectParameters["problem_data"]["domain_size"].GetInt())

        # Construct the solver (main setting methods are located in the solver_module)
        import python_solvers_wrapper_contact_structural
        self.solver = python_solvers_wrapper_contact_structural.CreateSolver(self.main_model_part, self.ProjectParameters)

        # Add variables (always before importing the model part) (it must be integrated in the ImportModelPart)
        # If we integrate it in the model part we cannot use combined solvers
        self.solver.AddVariables()

        # Read model_part (note: the buffer_size is set here) (restart can be read here)
        self.solver.ImportModelPart()

        # Add dofs (always after importing the model part) (it must be integrated in the ImportModelPart)
        # If we integrate it in the model part we cannot use combined solvers
        self.solver.AddDofs()

        # Build sub_model_parts or submeshes (rearrange parts for the application of custom processes)
        self.Model = KratosMultiphysics.Model()
        self.Model.AddModelPart(self.main_model_part)

        # Obtain the list of the processes to be applied
        self.list_of_processes = process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["constraints_process_list"])
        self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["loads_process_list"])
        if (self.ProjectParameters.Has("list_other_processes") == True):
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["list_other_processes"])
        if (self.ProjectParameters.Has("json_check_process") == True):
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["json_check_process"])
        if (self.ProjectParameters.Has("json_output_process") == True):
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["json_output_process"])
        if (self.ProjectParameters.Has("contact_process_list") == True): # NOTE: Always add the contact processes the last one (to avoid problems imposing displacements)
            self.list_of_processes += process_factory.KratosProcessFactory(self.Model).ConstructListOfProcesses(self.ProjectParameters["contact_process_list"])

        for process in self.list_of_processes:
            process.ExecuteInitialize()

        # ### START SOLUTION ####

        self.computing_model_part = self.solver.GetComputingModelPart()
        self.solver.AddProcessesList(self.list_of_processes)

        # ### Output settings start ####
        self.problem_path = os.getcwd()
        self.problem_name = self.ProjectParameters["problem_data"]["problem_name"].GetString()

        # ### Output settings start ####
        self.output_post = self.ProjectParameters.Has("output_configuration")
        if (self.output_post == True):
            if (self.parallel_type == "OpenMP"):
                from gid_output_process import GiDOutputProcess
                output_settings = self.ProjectParameters["output_configuration"]
                self.gid_output = GiDOutputProcess(self.computing_model_part,
                                                   self.problem_name,
                                                   output_settings)
            elif (self.parallel_type == "MPI"):
                from gid_output_process_mpi import GiDOutputProcessMPI
                output_settings = self.ProjectParameters["output_configuration"]
                self.gid_output = GiDOutputProcessMPI(self.computing_model_part,
                                                      self.problem_name,
                                                      output_settings)
            self.gid_output.ExecuteInitialize()

        # Sets strategies, builders, linear solvers, schemes and solving info, and fills the buffer
        self.solver.Initialize()
        self.solver.SetEchoLevel(0) # Avoid to print anything

        if (self.output_post == True):
            self.gid_output.ExecuteBeforeSolutionLoop()
            self.solver.AddPostProcess(self.gid_output)

    def Solve(self):
        for process in self.list_of_processes:
            process.ExecuteBeforeSolutionLoop()

        # #Stepping and time settings (get from process info or solving info)
        # Delta time
        delta_time = self.ProjectParameters["problem_data"]["time_step"].GetDouble()
        # Start step
        self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] = 0
        # Start time
        time = self.ProjectParameters["problem_data"]["start_time"].GetDouble()
        # End time
        end_time = self.ProjectParameters["problem_data"]["end_time"].GetDouble()

        # Solving the problem (time integration)
        while(time <= end_time):
            time = time + delta_time
            self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] += 1
            self.main_model_part.CloneTimeStep(time)

            for process in self.list_of_processes:
                process.ExecuteInitializeSolutionStep()

            if (self.output_post == True):
                self.gid_output.ExecuteInitializeSolutionStep()

            self.solver.Clear()
            self.solver.Solve()

            if (self.output_post == True):
                self.gid_output.ExecuteFinalizeSolutionStep()

            for process in self.list_of_processes:
                process.ExecuteFinalizeSolutionStep()

            for process in self.list_of_processes:
                process.ExecuteBeforeOutputStep()

            if (self.output_post == True):
                if self.gid_output.IsOutputStep():
                    self.gid_output.PrintOutput()

            for process in self.list_of_processes:
                process.ExecuteAfterOutputStep()

        if (self.output_post == True):
            self.gid_output.ExecuteFinalize()

        for process in self.list_of_processes:
            process.ExecuteFinalize()
