import datetime
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numpy import number
from numpy.lib.user_array import container

import random
import copy
import itertools

import The_Model as Model
from pandas import Timestamp, DateOffset
import csv
from collections import OrderedDict
import gurobipy as gp
from gurobipy import GRB
import re


class The_Solution_OR_TM:
    def __init__(self, problem, allocated_work_instructions=None, name="Main"):
        self.problem = problem
        if allocated_work_instructions is None:
            allocated_work_instructions = []
        self.allocated_work_instructions = allocated_work_instructions
        self.objective_function = sys.maxsize  # The current objective function
        self.best_found_objective_function = sys.maxsize  # The best found objective function
        self.name = name


    # Print solution
    def print_allocations(self):
        print("=" * 80)
        num_of_sc = self.problem.sc_number
        max_sc_name_length = max(len(str(sc)) for sc in self.problem.scs)

        for i, wi_list in enumerate(self.allocated_work_instructions):
            sc_name = str(self.problem.scs[i])

            sc_name_formatted = sc_name.ljust(max_sc_name_length)

            if wi_list is not None and len(wi_list) > 0:
                wi_str = ' '.join(str(wi) for wi in wi_list)
            else:
                wi_str = "No work instructions assigned"
            print(f"{sc_name_formatted} [ {wi_str} ]")
        print("=" * 80)
        print()

    def print_allocations_latex(self):
        def escape_latex(s):
            return s.replace('_', r'\_')

        data = []
        num_of_sc = self.problem.sc_number

        for i, wi_list in enumerate(self.allocated_work_instructions):
            sc_name = escape_latex(str(self.problem.scs[i]))

            if wi_list is not None and len(wi_list) > 0:
                wi_str = ' \\ '.join(escape_latex(str(wi)) for wi in wi_list)
            else:
                wi_str = "No work instructions assigned"

            data.append([sc_name, wi_str])

        headers = ['Straddle Carrier', 'Assigned Work Instructions']

        table = "\\begin{tabular}{ll}\n\\toprule\n"
        table += " & ".join(headers) + " \\\\\n\\midrule\n"

        for row in data:
            table += " & ".join(row) + " \\\\\n"

        table += "\\bottomrule\n\\end{tabular}"

        print(table + "\n\n")

    def print_WI_times(self):
        for i in self.problem.wis:
            print(
                f"{i} -- {i.name} || {i.earliest_move_start_time} || {i.latest_move_start_time} "
                f"||{i.earliest_move_end_time} || {i.latest_move_end_time}")

    def print_WI_solution_times_latex(self):
        def escape_latex(s):
            return s.replace('_', r'\_')

        headers = [
            "Work Instruction",
            "Earliest Start",
            "Latest Start",
            "Earliest End",
            "Latest End",
            "Planned Start",
            "Planned End"
        ]

        data = []
        for i in self.problem.wis:
            data.append([
                escape_latex(str(i)),
                str(i.earliest_move_start_time),
                str(i.latest_move_start_time),
                str(i.earliest_move_end_time),
                str(i.latest_move_end_time),
                str(i.planned_start_time),
                str(i.planned_end_time)
            ])

        table = "\\resizebox{\\textwidth}{!}{%\n"
        table += "\\begin{tabular}{l l l l l l l}\n\\toprule\n"
        table += " & ".join(headers) + " \\\\\n\\midrule\n"

        for row in data:
            table += " & ".join(row) + " \\\\\n"

        table += "\\bottomrule\n\\end{tabular}}\n"

        print(table + "\n\n")

    def print_WI_solution_times_latex_90degree(self):
        def escape_latex(s):
            return s.replace('_', r'\_')

        headers = [
            "Work Instruction",
            "Earliest Start",
            "Latest Start",
            "Earliest End",
            "Latest End",
            "Planned Start",
            "Planned End"
        ]

        data = []
        for i in self.problem.wis:
            data.append([
                escape_latex(str(i)),
                str(i.earliest_move_start_time),
                str(i.latest_move_start_time),
                str(i.earliest_move_end_time),
                str(i.latest_move_end_time),
                str(i.planned_start_time),
                str(i.planned_end_time)
            ])

        table = "\\begin{center}\n"
        table += "\\rotatebox{90}{%\n"
        table += "\\resizebox{\\textheight}{!}{%\n"
        table += "\\begin{tabular}{l l l l l l l}\n\\toprule\n"
        table += " & ".join(headers) + " \\\\\n\\midrule\n"

        for row in data:
            table += " & ".join(row) + " \\\\\n"

        table += "\\bottomrule\n\\end{tabular}}}\n"
        table += "\\end{center}"

        print(table + "\n\n")

    def copy_allocated_work_instructions(self):
        new_list = []
        for inner_list in self.allocated_work_instructions:
            new_inner_list = []
            for object in inner_list:
                new_inner_list.append(object)
            # new_inner_list = copy.copy(inner_list)
            new_list.append(new_inner_list)
        return new_list

    def the_greedy(self):

        num_of_SCs = self.problem.sc_number

        objective = 0
        # self.objective_elements[0] = 0
        # self.objective_elements[1] = 0

        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None

        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.planned_time_is_calculated = False

        self.allocated_work_instructions = []
        is_first = []
        for i in range(0, num_of_SCs):
            self.allocated_work_instructions.append([])
            is_first.append(True)

        num_of_WIs = self.problem.wis_number

        i = 0
        while i < num_of_WIs:

            wi = self.problem.wis[i]

            list_of_SCs = self.problem.scs.copy()

            list_of_SCs.sort(
                key=lambda x: (x.last_WI_finish_time + timedelta(seconds=x.last_position.duration(wi.start_position))))
            index = 0  # greedy manner

            current_SC = list_of_SCs[index]

            if current_SC.last_WI is None:
                previous_location = current_SC.start_position
            else:
                previous_location = current_SC.last_position

            time_Needed_in_Seconds = previous_location.duration(wi.start_position)

            previous_end_time = current_SC.last_WI_finish_time

            start_time_wi = previous_end_time + timedelta(seconds=time_Needed_in_Seconds)
            wi.planned_start_time = start_time_wi
            # travel_distance = wi.start_position.distance_to(wi.end_position)
            travel_time = wi.start_position.duration(wi.end_position)
            wi.planned_end_time = start_time_wi + timedelta(seconds=travel_time)
            wi.sc_at_initial_at = wi.planned_start_time
            wi.sc_at_final_at = wi.planned_end_time

            if wi.planned_start_time <= wi.earliest_move_start_time:
                wi.planned_start_time = wi.earliest_move_start_time
                wi.planned_end_time = wi.planned_start_time + timedelta(seconds=travel_time)
                wi.sc_at_final_at = wi.planned_end_time

            if wi.planned_start_time > wi.latest_move_start_time:
                print("Violation A**********************")
                print(wi)
                return False

            if wi.planned_end_time <= wi.earliest_move_end_time:
                wi.planned_end_time = wi.earliest_move_end_time

            if wi.planned_end_time > wi.latest_move_end_time:
                print("Violation B**********************")
                print(wi)
                return False

            wi.sc = current_SC
            if current_SC.last_WI is None:
                wi.sc_start_for = current_SC.start_time
            else:
                wi.sc_start_for = current_SC.last_WI.planned_end_time
            objective += time_Needed_in_Seconds
            wi.planned_time_is_calculated = True
            sc_index = self.problem.scs.index(current_SC)
            self.allocated_work_instructions[sc_index].append(wi)

            current_SC.last_position = wi.end_position
            current_SC.current_position = wi.end_position
            current_SC.last_WI_finish_time = wi.planned_end_time
            current_SC.last_WI = wi
            i = i + 1

        self.objective_function = objective
        return True

    def local_search(self, time_limit=180, time_at_start=None):
        if time_at_start is None:
            time_at_start = time.time()
        number_of_SCs = self.problem.sc_number
        has_improvement = False


        for i in range(0, number_of_SCs):
            for j in range(0, number_of_SCs):
                if i != j:
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2exchange_best_found(i, j, time_limit, time_at_start)
                    if was_improvement:
                        has_improvement = True
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement

        for i in range(0, number_of_SCs):
            for j in range(0, number_of_SCs):
                if i != j:
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2opt_2route_best_found(i, j, time_limit, time_at_start)
                    if was_improvement:
                        has_improvement = True
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement

        for i in range(0, number_of_SCs):
            for j in range(0, number_of_SCs):
                if i != j:
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2relocate_best_found(i, j, time_limit, time_at_start)
                    if was_improvement:
                        has_improvement = True
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement

        for i in range(0, number_of_SCs):
           current_time = time.time() - time_at_start
           if current_time > time_limit:
               return has_improvement
           was_improvement = self.LS_2opt_best_found(i, time_limit, time_at_start)
           if was_improvement:
               # print("Was improvement 2 opt")
               has_improvement = True

        return has_improvement

    def local_search_for_test(self, operator_list, time_limit=3600, time_at_start=None):
        if time_at_start is None:
            time_at_start = time.time()
        number_of_SCs = self.problem.sc_number
        has_improvement = False

        for operator in operator_list:
            if operator == "2Exchange":
                for i in range(0, number_of_SCs):
                    for j in range(0, number_of_SCs):
                        if i != j:
                            current_time = time.time() - time_at_start
                            if current_time > time_limit:
                                return has_improvement
                            was_improvement = self.LS_2exchange_best_found(i, j)
                            if was_improvement:
                                has_improvement = True

            if operator == "2Relocate":
                for i in range(0, number_of_SCs):
                    for j in range(0, number_of_SCs):
                        if i != j:
                            current_time = time.time() - time_at_start
                            if current_time > time_limit:
                                return has_improvement
                            was_improvement = self.LS_2relocate_best_found(i, j)
                            if was_improvement:
                                has_improvement = True

            if operator == "2Opt2Route":
                for i in range(0, number_of_SCs):
                    for j in range(0, number_of_SCs):
                        if i != j:
                            current_time = time.time() - time_at_start
                            if current_time > time_limit:
                                return has_improvement
                            was_improvement = self.LS_2opt_2route_best_found(i, j)
                            if was_improvement:
                                has_improvement = True

            if operator == "2Opt":
                for i in range(0, number_of_SCs):
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2opt_best_found(i)
                    if was_improvement:
                        has_improvement = True

        return has_improvement

    def local_search_for_VND(self, k, time_limit=3600, time_at_start=None):
        if time_at_start is None:
            time_at_start = time.time()
        number_of_SCs = self.problem.sc_number
        has_improvement = False

        if k == 3:
            for i in range(0, number_of_SCs):
                for j in range(0, number_of_SCs):
                    if i != j:
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement
                        was_improvement = self.LS_2exchange_best_found(i, j, time_limit, time_at_start)
                        if was_improvement:
                            has_improvement = True
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement
        elif k == 1:
            for i in range(0, number_of_SCs):
                for j in range(0, number_of_SCs):
                    if i != j:
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement
                        was_improvement = self.LS_2relocate_best_found(i, j, time_limit, time_at_start)
                        if was_improvement:
                            has_improvement = True
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement
        elif k == 2:
            for i in range(0, number_of_SCs):
                for j in range(0, number_of_SCs):
                    if i != j:
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement
                        was_improvement = self.LS_2opt_2route_best_found(i, j, time_limit, time_at_start)
                        if was_improvement:
                            has_improvement = True
                        current_time = time.time() - time_at_start
                        if current_time > time_limit:
                            return has_improvement

        else:
            for i in range(0, number_of_SCs):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    return has_improvement
                was_improvement = self.LS_2opt_best_found(i, time_limit, time_at_start)
                if was_improvement:
                    has_improvement = True
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    return has_improvement

        return has_improvement

    def LS_2exchange_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = The_Solution_OR_TM(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break
            for index2 in range(number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break
                new_solution = The_Solution_OR_TM(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                new_solution.allocated_work_instructions[sc_index1][index1], \
                    new_solution.allocated_work_instructions[sc_index2][index2] = \
                    new_solution.allocated_work_instructions[sc_index2][index2], \
                        new_solution.allocated_work_instructions[sc_index1][index1]

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        return False

    def LS_2relocate_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        if number_of_WIs1 == 1:
            return False
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = The_Solution_OR_TM(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break
            for index2 in range(number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break
                new_solution = The_Solution_OR_TM(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                WI_to_relocate = new_solution.allocated_work_instructions[sc_index1][index1]
                new_solution.allocated_work_instructions[sc_index1].remove(WI_to_relocate)
                new_solution.allocated_work_instructions[sc_index2].insert(index2, WI_to_relocate)

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        return False

    def LS_2opt_2route_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = The_Solution_OR_TM(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(1, number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break
            for index2 in range(1, number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break
                new_solution = The_Solution_OR_TM(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                temp = new_solution.allocated_work_instructions[sc_index1][index1:]
                new_solution.allocated_work_instructions[sc_index1][index1:] = new_solution.allocated_work_instructions[
                                                                                   sc_index2][index2:]
                new_solution.allocated_work_instructions[sc_index2][index2:] = temp

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        return False

    def LS_2opt_best_found(self, sc_index1, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        best_solution = The_Solution_OR_TM(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(0, number_of_WIs1 - 1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break
            for index2 in range(index1 + 2, number_of_WIs1):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break
                new_solution = The_Solution_OR_TM(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
                new_solution.allocated_work_instructions[sc_index1] = (
                        new_solution.allocated_work_instructions[sc_index1][:index1] +
                        new_solution.allocated_work_instructions[sc_index1][index1:index2][::-1] +
                        new_solution.allocated_work_instructions[sc_index1][index2:])

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        return False

    def real_world_solution(self):

        for wi in self.problem.wis:
            if wi.real_assignment is None:
                raise ValueError(f"Cannot calculated with {wi.name} has no real assignment")


        num_of_SCs = self.problem.sc_number

        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None

        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.planned_time_is_calculated = False

        self.allocated_work_instructions = []
        is_first = []
        for i in range(0, num_of_SCs):
            self.allocated_work_instructions.append([])
            is_first.append(True)


        for sc in self.problem.scs:
            index = sc.index
            self.allocated_work_instructions[index] = sc.real_wis

        objective = self.calculation_of_obj()
        self.objective_function = objective

        return objective


    def calculation_of_obj(self, to_beat=None):


        objective = 0


        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None

        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.planned_time_is_calculated = False

        for i, sc in enumerate(self.problem.scs):
            if not self.allocated_work_instructions[i]:
                sc.next_WI = None
            else:
                sc.next_WI = self.allocated_work_instructions[i][0]

        for straddle_carrier_index, straddleC in enumerate(self.allocated_work_instructions):
            for order, wi in enumerate(straddleC):
                wi.assignment_index = (straddle_carrier_index, order)

        remaining_wi = True

        while remaining_wi:
            remaining_wi = False
            for current_SC in self.problem.scs:
                wi = current_SC.next_WI

                if wi is None:
                    continue

                remaining_wi = True

                if current_SC.last_WI is None:
                    previous_location = current_SC.start_position
                else:
                    previous_location = current_SC.last_position

                time_Needed_in_Seconds = previous_location.duration(wi.start_position)

                previous_end_time = current_SC.last_WI_finish_time

                start_time_wi = previous_end_time + timedelta(seconds=time_Needed_in_Seconds)
                wi.planned_start_time = start_time_wi
                travel_time = wi.start_position.duration(wi.end_position)
                wi.planned_end_time = start_time_wi + timedelta(seconds=travel_time)
                wi.sc_at_initial_at = wi.planned_start_time
                wi.sc_at_final_at = wi.planned_end_time

                if wi.planned_start_time < wi.earliest_move_start_time:
                    wi.planned_start_time = wi.earliest_move_start_time
                    wi.planned_end_time = wi.planned_start_time + timedelta(seconds=travel_time)
                    wi.sc_at_final_at = wi.planned_end_time

                if wi.planned_start_time > wi.latest_move_start_time:
                    # print("Violation A**********************")
                    # print(wi)
                    return False

                if wi.planned_end_time < wi.earliest_move_end_time:
                    wi.planned_end_time = wi.earliest_move_end_time

                if wi.planned_end_time > wi.latest_move_end_time:
                    # print("Violation B**********************")
                    # print(wi)
                    return False

                wi.sc = current_SC
                if current_SC.last_WI is None:
                    wi.sc_start_for = current_SC.start_time
                else:
                    wi.sc_start_for = current_SC.last_WI.planned_end_time
                objective += time_Needed_in_Seconds  # Overall empty travel distances
                wi.planned_time_is_calculated = True


                if to_beat is not None and objective > to_beat:
                    return False

                current_SC.last_position = wi.end_position
                current_SC.last_WI_finish_time = wi.planned_end_time
                current_SC.last_WI = wi
                if wi != self.allocated_work_instructions[wi.assignment_index[0]][-1]:
                    current_SC.next_WI = self.allocated_work_instructions[wi.assignment_index[0]][
                        wi.assignment_index[1] + 1]
                else:
                    current_SC.next_WI = None

        self.objective_function = objective
        return True


class VND:
    shift_file = "V4_shifts_from_telemetry.parquet"
    wi_file = "V4_jobinstructions_from_telemetry.parquet"

    def __init__(self, shift_id, planning_interval, shift_file, wi_file):
        self.vnd_problem = Model.Problem(shift_id, planning_interval)
        self.vnd_problem.read_file(shift_file, wi_file)
        self.vnd_solution = The_Solution_OR_TM(self.vnd_problem)

    def run(self, kmax=4, time_limit=180):
        start_vnd_time = time.time()
        self.vnd_solution.the_greedy()
        greedy_value = self.vnd_solution.objective_function
        best_solution = The_Solution_OR_TM(self.vnd_problem, name="Best VND")
        best_solution.allocated_work_instructions = self.vnd_solution.copy_allocated_work_instructions()
        best_solution.objective_function = greedy_value

        k = 1
        while k <= kmax:
            self.vnd_solution.local_search_for_VND(k, time_limit=time_limit, time_at_start=start_vnd_time)
            (best_solution.allocated_work_instructions, best_solution.objective_function, k) = (
                self.neighborhood_change(best_solution, k))
        end_vnd_time = time.time() - start_vnd_time

        self.vnd_solution.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
        self.vnd_solution.objective_function = best_solution.objective_function
        vnd_result = best_solution.objective_function

        improvement_rate_greedy_to_VND = ((greedy_value - vnd_result) / greedy_value) * 100
        real_time_value = self.vnd_solution.problem.real_world_time
        improvement_real_world = ((real_time_value - vnd_result) / real_time_value) * 100

        return greedy_value, improvement_rate_greedy_to_VND, improvement_real_world, vnd_result, end_vnd_time

    def neighborhood_change(self, best_sol, k):
        new_objective = self.vnd_solution.objective_function
        best_objective = best_sol.objective_function
        if new_objective < best_objective:
            best_sol.allocated_work_instructions = self.vnd_solution.copy_allocated_work_instructions()
            best_sol.objective_function = new_objective
            k = 1
        else:
            k += 1
        return best_sol.allocated_work_instructions, best_sol.objective_function, k


