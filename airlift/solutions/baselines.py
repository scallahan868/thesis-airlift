import random

import networkx as nx
from networkx import NetworkXNoPath
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID, ObservationHelper
from airlift.solutions import Solution
from airlift.envs import PlaneState
from pyomo.opt import SolverFactory

from milptools import solve_multiple_aircraft

REPLANINTERVAL = 100


# Random agent which chooses only valid actions
class RandomAgent(Solution):
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)
        self._action_helper = ActionHelper(np_random=self._np_random)

    def policies(self, obs, dones, infos=None):
        return self._action_helper.sample_valid_actions(observation=obs)


class ShortestPath(Solution):
    def __init__(self):
        super().__init__()

        self.cargo_delivered = None
        self._full_delivery_paths = None
        self.multidigraph = None
        self.multi_view = None
        self.cargo_assignments = None
        self.path = None
        self.plane_graph = None
        self.view = None
        self.plane_types = []

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)

        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.view = {}
        self._full_delivery_paths = {}
        self.cargo_delivered = {a: [] for a in self.agents}

    def policies(self, obs, dones, infos=None):
        actions = {}
        state = self.get_state(obs)

        # Build views in-case of malfunctions
        self.build_multidigraph_view(state)
        self.build_agent_views(obs, state)

        # Create full delivery path for all active cargo
        self.get_initial_full_delivery_path(state)
        pending_cargo = self.get_active_cargo_from_state(state)

        # Create our cargo bins
        cargo_bins = self.separate_cargo_by_location_and_destination(pending_cargo)

        # Generate an initial plan of action
        self.get_initial_agent_actions(obs, state, dones, pending_cargo, cargo_bins)
        actions.update(self.plan(obs))

        # Use ActionHelper and ObservationHelper to update the action again in-case of non-valid entries.
        self.check_for_valid_actions(obs, state, actions)

        return actions

    def plan(self, obs):
        state = self.get_state(obs)
        actions = {a: None for a in self.agents}

        for a in self.agents:

            if oh.needs_orders(obs[a]):
                actions[a] = {"priority": 1,
                              "cargo_to_load": [],
                              "cargo_to_unload": [],
                              "destination": NOAIRPORT_ID}

                if self.path[a]:
                    next_destination = self.path[a][0]

                    if obs[a]["current_airport"] == next_destination:
                        self.path[a].pop(0)
                        if self.path[a]:
                            next_destination = self.path[a][0]

                    actions[a]["destination"] = next_destination
                ca = oh.get_active_cargo_info(state, self.cargo_assignments[a])
                if ca is not None:
                    if ca[0].id in obs[a]["cargo_onboard"]:
                        for cargo in ca:
                            if cargo.destination == obs[a]['current_airport'] or len(self.path[a]) == 0:
                                actions[a]["cargo_to_unload"].append(cargo.id)
                                # self.cargo_assignments[a].remove(cargo) <--Breaks things for some reason..
                                # Just setting it to None seems to fix it, but can't even set it to None outside the loop...
                                self.cargo_assignments[a] = None

                    elif ca[0].id in obs[a]['cargo_at_current_airport']:
                        assert ca[0].destination != obs[a]['current_airport']
                        for cargo in ca:
                            if cargo.id not in self.cargo_delivered[a]:
                                actions[a]["cargo_to_load"].append(cargo.id)
                                self.cargo_delivered[a].append(cargo.id)

        return actions

    def build_agent_views(self, obs, state):
        if not self.plane_types:
            for a in self.agents:
                if obs[a]['plane_type'] not in self.plane_types:
                    self.plane_types.append(obs[a]['plane_type'])

        # Add the subgraph view for each plane_type
        for plane_type in self.plane_types:
            self.view[plane_type] = nx.subgraph_view(state["route_map"][plane_type],
                                                     filter_edge=self.filter_edge)

    def build_multidigraph_view(self, state):
        self.multidigraph = oh.get_multidigraph(state)
        self.multi_view = nx.subgraph_view(self.multidigraph, filter_edge=self.filter_multi_graph_edge)

    def get_initial_full_delivery_path(self, state):
        assert all(c.location != c.destination for c in state["active_cargo"])
        for c in state["active_cargo"]:
            if c.location != NOAIRPORT_ID and c.id not in self._full_delivery_paths:
                try:
                    self._full_delivery_paths[c.id] = nx.shortest_path(self.multi_view, c.location, c.destination,
                                                                       weight="cost")[1:]
                except nx.NetworkXNoPath as e:
                    continue

    def get_active_cargo_from_state(self, state):
        pending_cargo = [c for c in state["active_cargo"] if
                         c.id not in self.cargo_assignments.values() and c.is_available == 1]
        return pending_cargo

    def separate_cargo_by_location_and_destination(self, cargo_list):

        cargo_bins = {}

        for cargo_item in cargo_list:
            location = cargo_item.location
            destination = cargo_item.destination
            key = (location, destination)

            if key not in cargo_bins:
                cargo_bins[key] = []
            if cargo_item.is_available:
                cargo_bins[key].append(cargo_item)

        return cargo_bins

    def select_cargo(self, cargo_bins):
        keys_list = list(cargo_bins.keys())
        random_key_from_list = tuple(self._np_random.choice(keys_list))
        cargo_info = cargo_bins[random_key_from_list]
        return random_key_from_list, cargo_info

    def get_initial_agent_actions(self, obs, state, dones, pending_cargo, cargo_bins):
        active_cargo_ids = [c.id for c in pending_cargo]
        for a in self.agents:
            plane_type = obs[a]['plane_type']
            self.plane_graph = state["route_map"][plane_type]

            # Create a copy of the cargo assignments and remove anything that shouldn't be there
            if self.cargo_assignments[a] is not None:
                cargo_copy = list(self.cargo_assignments[a])
                for cargo in cargo_copy:
                    if cargo not in active_cargo_ids:
                        self.cargo_assignments[a].remove(cargo)

            if dones[a]:
                continue

            if pending_cargo and self.cargo_assignments[a] is None:
                # Select cargo info and return the randomly selected key (loc, dest) pair.
                key, cargo_info = self.select_cargo(cargo_bins)
                if cargo_info[0].id not in self.cargo_delivered[a]:
                    if cargo_info[0].location != NOAIRPORT_ID:
                        if cargo_info[0].id in self._full_delivery_paths:
                            full_delivery_path = self._full_delivery_paths[cargo_info[0].id]
                        else:
                            try:
                                full_delivery_path = nx.shortest_path(self.multi_view, cargo_info[0].location,
                                                                      cargo_info[0].destination,
                                                                      weight="cost")  # [1:]
                            except NetworkXNoPath as e:
                                continue
                        try:
                            if full_delivery_path:
                                if not self.view[plane_type].has_edge(cargo_info[0].location, full_delivery_path[0]
                                                                      ):
                                    continue
                                path = oh.get_lowest_cost_path(self.view[plane_type], obs[a]["current_airport"],
                                                               cargo_info[0].location,
                                                               obs[a]["plane_type"])

                                while full_delivery_path and self.view[plane_type].has_edge(path[-1],
                                                                                            full_delivery_path[0],
                                                                                            ):
                                    path.append(full_delivery_path.pop(0))
                                self.path[a] = path

                                # Get the airplanes max carrying capacity and assign it cargo
                                max_airplane_weight = obs[a]['max_weight']
                                num_cargo_assigned = 0
                                assigned_cargo = []
                                cargo_info_copy = list(cargo_info)
                                for cargo in cargo_info_copy:
                                    assigned_cargo.append(cargo.id)
                                    pending_cargo.remove(cargo)
                                    num_cargo_assigned += 1
                                    cargo_info.remove(cargo)

                                    if num_cargo_assigned == max_airplane_weight:
                                     break

                                # Delete the key value if there are no more cargo in this bin, so we don't select it anymore
                                # during the next iteration
                                if not cargo_info:
                                    del cargo_bins[key]

                                self.cargo_assignments[a] = assigned_cargo

                                # If there are no more bins to assign, break out of the loop
                                if not cargo_bins:
                                    break

                        except NetworkXNoPath as e:
                            continue

    def create_path_and_update_action(self, obs, a, cargo_info, full_delivery_path, actions):
        plane_type = obs[a]['plane_type']

        # If we can't make any progress
        if not self.view[plane_type].has_edge(cargo_info[0].location, full_delivery_path[0]):
            return False

        path = oh.get_lowest_cost_path(self.view[plane_type], obs[a]["current_airport"],
                                       cargo_info[0].destination,
                                       obs[a]["plane_type"])

        while full_delivery_path and self.view[plane_type].has_edge(path[-1], full_delivery_path[0]):
            path.append(full_delivery_path.pop(0))

        self.path[a] = path
        actions[a].destination = self.path[a].pop()
        actions[a]['cargo_to_unload'].add(cargo_info.id)

    def get_full_delivery_path_by_cargo_info(self, obs, cargo_info):
        self._full_delivery_paths[cargo_info[0].id] = nx.shortest_path(self.multi_view,
                                                                       obs['current_airport'],
                                                                       cargo_info[0].destination,
                                                                       weight="cost")[1:]

        for cargo in cargo_info:
            if cargo is not None:
                self._full_delivery_paths[cargo.id] = self._full_delivery_paths[cargo_info[0].id]

        return self._full_delivery_paths[cargo_info[0].id]

    def check_for_valid_actions(self, obs, state, actions):
        for a in self.agents:
            valid = ActionHelper.is_action_valid(actions[a], obs[a])
            if not valid[0]:
                cargo_info = ObservationHelper.get_cargo_objects(state, obs[a]['cargo_onboard'])
                if cargo_info:
                    if cargo_info[0] is not None:
                        try:
                            full_delivery_path = self.get_full_delivery_path_by_cargo_info(obs[a], cargo_info)
                        except NetworkXNoPath as e:
                            continue

                        if full_delivery_path:
                            try:
                                if not self.create_path_and_update_action(obs, a, cargo_info, full_delivery_path,
                                                                          actions):
                                    continue

                            except NetworkXNoPath as e:
                                continue

    def filter_edge(self, u, v):
        """Filter DiGraph, used for Airplane Types graphs"""
        return self.plane_graph[u][v]["mal"] == 0

    # Filter a multidigraph

    def filter_multi_graph_edge(self, u, v, key):
        """Filter the MultiDiGraph (created from collection of DiGraphs)"""
        return self.multidigraph[u][v][key]['mal'] == 0

    # Check to see if the subgraph edges/nodes exist in the multigraph for our assertion
    def is_subgraph_of_multigraph(self, subgraph, multigraph):
        """Not utilized, but can be used to assert that the newly created views are subgraphs of the multi di graph"""
        if not set(subgraph.nodes).issubset(set(multigraph.nodes)):
            return False

        for u, v, data in subgraph.edges(data=True):
            if not multigraph.has_edge(u, v):
                return False
        return True


def make_cargo_event_list(MILP_event_list):
    new_event_list = []

    i = 0
    while i < len(MILP_event_list):
        event = MILP_event_list[i]
        airportid =  event["airportid"]
        eventType = event["eventType"]
        cargoId = event["cargoId"]
        if eventType == "start":
            if i > 0:
                assert MILP_event_list[i-1]["eventType"] == "end"
                assert airportid == MILP_event_list[i-1]["airportid"] # This could be relaxed
        elif eventType == "end":
            assert airportid == MILP_event_list[i-1]["airportid"] # This could be relaxed
        else: # Must be cargo event
            if eventType == "pickup":
                cargo_to_load = [cargoId]
                cargo_to_unload = []
            elif eventType == "drop":
                cargo_to_load = []
                cargo_to_unload = [cargoId]
            else:
                assert False, "Unrecognized event"

            # Update last cargo event
            if new_event_list and airportid == new_event_list[-1]["airportid"]:
                new_event_list[-1]["load"].extend(cargo_to_load)
                new_event_list[-1]["unload"].extend(cargo_to_unload)
            # Make a new cargo event
            else:
                new_event_list.append({
                    "airportid": airportid,
                    "load": cargo_to_load,
                    "unload": cargo_to_unload
                })

        i += 1

    return new_event_list



class AirplaneSequencer:
    # TODO: Update path if route goes offline
    def __init__(self):
        self._event_list = []
        # Indicates the next event which needs to occur. Will be None if the current event list has been completed (or empty)
        self._next_event_index = None
        self._path = None

    # Get the next event
    @property
    def _next_event(self):
        if self._next_event_index is None:
            return None
        else:
            return self._event_list[self._next_event_index]

    # Move to the next event
    def _advance_event_index(self):
        self._next_event_index += 1
        if self._next_event_index >= len(self._event_list):
            self._next_event_index = None

    # Generates an action for the given ariplane based on its event list
    def airplane_policy(self, obs):
        state = obs["globalstate"]

        # We will only assign a new action if airplane is ready for next orders and we have more events to process
        if oh.needs_orders(obs) and self._next_event is not None:
            # The next airport is the one we are at or the one we are in flight to
            next_airport = obs["current_airport"] if obs["destination"] == NOAIRPORT_ID else obs["destination"]

            # Let's start building the next action
            action = {"priority": 1,  # All airplanes are priority 1
                      "cargo_to_load": [],
                      "cargo_to_unload": [],
                      "destination": NOAIRPORT_ID}

            # Will we be arriving at the airport for the next event?
            if self._next_event["airportid"] == next_airport:
                # Indicate the cargo to load/unload associated with next event (once we arrive at the event airport)
                action["cargo_to_load"] = self._next_event["load"]
                action["cargo_to_unload"] = self._next_event["unload"]

                # We can start processing the next event (if there is one)
                self._advance_event_index()

            # If there are more events to process
            if self._next_event is not None:
                # If we don't have a path yet, set a path to the next event (note if we are already at the next event's airport, this will be an empty list)
                if not self._path:
                    path = oh.get_lowest_cost_path(state,
                                                   next_airport,
                                                   self._next_event["airportid"],
                                                   obs["plane_type"])
                    # The 1st airport in the path should be the airport we are going to (or already at)
                    assert path[0] == next_airport
                    self._path = path[1:]

                # Set the next airport according to the path (if there is a path)
                if self._path:
                    action["destination"] = self._path.pop(0)

            return action
        else:
            # We already have an action pending. Leave it as-is.
            return None

    # Which is the last airport in the current sequence. For planning additional events.
    @property
    def end_airport(self):
        if not self.event_list:
            return None
        else:
            return self.event_list[-1]["airport_id"]

    # This is the only way the event list should be updated
    def add_event_list(self, event_list):
        # If current event list is complete start at 1st element in new event list
        if self._next_event_index is None and event_list:
            self._next_event_index = len(self._event_list)

        self._event_list.extend(event_list)


# TODO: Handle disconnected routes
class MILP(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()
        self._sequencer = None
        self._end_events = None

        self._new_cargo = None
        self._steps_since_last_replan = None

        # See https://stackoverflow.com/questions/51371067/pyomo-list-available-solvers
        if SolverFactory('cplex').available():
            print("Using cplex solver")
            self._solver = 'cplex'
        elif SolverFactory('glpk').available():
            print("Using glpk solver")
            self._solver = 'glpk'
        else:
            raise Exception("Either the glpk or cplex solver must be installed")

    # Takes in an event list from the MILP, converts into a consolidated "cargo event list", and adds this to each airplane.
    def _update_events(self, aircraft_id_to_key, eventLists):
        for i, event_list in eventLists.items():
            cargo_event_list = make_cargo_event_list(event_list)
            self._sequencer[aircraft_id_to_key[i]].add_event_list(cargo_event_list)

            if event_list:
                end_event = event_list[-1]
                assert end_event["eventType"] == "end"
                self._end_events[aircraft_id_to_key[i]] = end_event

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

        self._sequencer = {a: AirplaneSequencer() for a in obs}
        self._end_events = {a: None for a in obs}
        aircraft_id_to_key, S, Z, f_a, eventLists = solve_multiple_aircraft(obs, solver=self._solver)
        self._update_events(aircraft_id_to_key, eventLists)

        self._new_cargo = []
        self._steps_since_last_replan = 0

    def policies(self, obs, dones, infos):
        state = list(obs.values())[0]["globalstate"]

        # Collect new cargo and periodically plan delivery for the new cargo that as accumulated since the last replan
        self._new_cargo.extend(state["event_new_cargo"])
        self._steps_since_last_replan += 1
        if self._steps_since_last_replan > REPLANINTERVAL:
            end_airports = {}
            end_times = {}
            for a, e in self._end_events.items():
                # The airplane has not been used yet
                if e is None:
                    end_airports[a] = obs[a]["current_airport"]
                    end_times[a] = 0
                # The airplane is/has been used - capture its end state info
                else:
                    end_airports[a] = e["airportid"]
                    end_times[a] = e["eventTime"]

            # Do planning for new dynamic cargo - essentially we will let the airplanes finish what they are doing, then will have them handle the new dynamic cargo.
            # We pass in their end info as the start info for the next round of deliveries.
            aircraft_id_to_key, S, Z, f_a, eventLists = solve_multiple_aircraft(obs, start_airports=end_airports, earliest_start_times=end_times, cargo=self._new_cargo, solver=self._solver)
            self._update_events(aircraft_id_to_key, eventLists)

            self._new_cargo = []
            self._steps_since_last_replan = 0

        # Build actions using the sequencers
        actions = {}
        for a in self.agents:
            actions[a] = self._sequencer[a].airplane_policy(obs[a])

        return actions