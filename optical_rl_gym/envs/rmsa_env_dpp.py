import gym
import copy
import math
import heapq
import logging
import functools
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from optical_rl_gym.utils import Service, Path
from .optical_network_env import OpticalNetworkEnv



class RMSADPPEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate',
                    'bit_rate_blocking_rate', 'episode_bit_rate_blocking_rate',
                    'failure', 'episode_failure',
                    'failure_slots','episode_failure_slots', 
                    'failure_disjointness','episode_failure_disjointness',
                    'length_comparison',
                    'length_wp',
                    'length_bp',
                    'hops_wp',
                    'hops_bp',
                    'number_slots_wp',
                    'number_slots_bp',
                    'initial_slots_wp',
                    'initial_slots_bp',
                    'average_external_fragmentation_wp',
                    'average_external_fragmentation_bp',
                    ]
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=100,
                 node_request_probabilities=None,
                 bit_rate_lower_bound=50,
                 bit_rate_higher_bound=200,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False,
                 reset=True):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection)#,
                         #k_paths=k_paths)
        assert 'modulations' in self.topology.graph
        # specific attributes for elastic optical networks
        
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound
        self.failure_counter = 0 
        self.failure_disjointness = 0 
        self.episode_failure_disjointness = 0
        self.episode_failure_counter = 0 

        self.length_wp = 0
        self.length_bp = 0
        self.hops_wp = 0
        self.hops_bp = 0
        self.working_slots_2 = 0
        self.backup_slots_2 = 0
        self.initial_slot_working_2 = 0
        self.initial_slot_backup_2 = 0


        self.tmp = 0
        self.curr_ext_fragmentation_wp = 0
        self.curr_ext_fragmentation_bp = 0
        

        # A parameter to give a negative reward if the chosen working path is longer than the backup path
        self.long = 0
        

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces (need to be multiplied ?)
        # Need to review the spaces
        self.actions_output = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1, self.k_paths + 1,
                                        self.num_spectrum_resources + 1),
                                       dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + 1,
                                                self.num_spectrum_resources + 1,self.k_paths + 1,
                                                self.num_spectrum_resources + 1),
                                               dtype=int)
        # Double the number space in order to incorporate the working/backup logic
        self.actions_taken = np.zeros((self.k_paths + 1,
                                       self.num_spectrum_resources + 1,
                                       self.k_paths + 1,
                                       self.num_spectrum_resources + 1), dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                               self.num_spectrum_resources + 1,
                                               self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int)

        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                                      self.num_spectrum_resources + self.reject_action))
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action: [int]):  # We can refuse the service if one of the two is not working
        working_path, initial_slot_working, backup_path, initial_slot_backup = action[0], action[1], action[2], action[3]
        self.actions_output[working_path, initial_slot_working, backup_path, initial_slot_backup] += 1  # Will need to modify the actions_output
        
        if ((working_path < self.k_paths) and (initial_slot_working < self.num_spectrum_resources) and
            (backup_path < self.k_paths) and (initial_slot_backup < self.num_spectrum_resources)) :
  
            #   print("Working path {}".format(self.k_shortest_paths[self.service.source, self.service.destination][working_path].node_list))
            #   print("BU path {}".format(self.k_shortest_paths[self.service.source, self.service.destination][backup_path].node_list))
            #self.length_wp = self.k_shortest_paths[self.service.source, self.service.destination][working_path]('length') # Check which of the k paths
            #self.length_bp = backup_path('length')
            #self.length_comparison = (self.length_wp/self.length_bp)/((self.length_wp/self.length_bp) + 1)
            
            
  
            if not self.is_disjoint(self.k_shortest_paths[self.service.source, self.service.destination][working_path],
                                      self.k_shortest_paths[self.service.source, self.service.destination][backup_path]):  # We check that the paths are disjoint
                self.service.accepted = False  # If they are not then no need to go through the process
                self.failure_disjointness += 1
                self.episode_failure_disjointness += 1
            else:
                working_slots = self.get_number_slots(
                    self.k_shortest_paths[self.service.source, self.service.destination][working_path]) # What does this line do ?
                

                self.logger.debug(
                    '{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id,
                                                                                                action, working_path,
                                                                                                initial_slot_working,
                                                                                                working_slots))
                if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][working_path],
                                      initial_slot_working, working_slots):
                      # self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][working_path],
                      #                     initial_slot_working, working_slots,self.k_shortest_paths[self.service.source, self.service.destination][backup_path],initial_slot_backup, backup_slots )
    
                      # usage for both paths
                    backup_slots = self.get_number_slots(
                        self.k_shortest_paths[self.service.source, self.service.destination][backup_path])
                    self.logger.debug('{} processing action {} path {} and initial slot {} for {} slots'.format(
                        self.service.service_id, action, backup_path, initial_slot_backup, backup_slots))
                      
                    if self.is_path_free(
                            self.k_shortest_paths[self.service.source, self.service.destination][backup_path],
                            initial_slot_backup, backup_slots):
                          
    
                        self._provision_path(
                            self.k_shortest_paths[self.service.source, self.service.destination][working_path],
                            initial_slot_working, working_slots,
                            self.k_shortest_paths[self.service.source, self.service.destination]
                            [backup_path], initial_slot_backup, backup_slots)
                        self.service.accepted = True
                        self.actions_taken[
                            working_path, initial_slot_working, backup_path, initial_slot_backup] += 1
                        

                        self.length_wp = self.service.route.length
                        self.length_bp = self.service.backup_route.length
                        self.hops_wp = self.service.route.hops
                        self.hops_bp = self.service.backup_route.hops
                        self.working_slots_2 = self.service.number_slots
                        self.backup_slots_2 = self.service.number_slots_backup
                        self.initial_slot_working_2 = self.service.initial_slot
                        self.initial_slot_backup_2 = self.service.initial_slot_backup
                        self.length_comparison = (self.length_wp/self.length_bp)/((self.length_wp/self.length_bp) + 1)

                        
                        if self.length_bp < self.length_wp:
                            self.long += 1

                        self._add_release(self.service)
                          
                          
                              
                    else:
                        self.service.accepted = False
    
                else:
                    self.service.accepted = False

        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1
            self.failure_counter += 1
            self.episode_failure_counter += 1 
            
    
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.service.bit_rate
        self.episode_bit_rate_requested += self.service.bit_rate
    
        self.topology.graph['services'].append(self.service)
    
        reward = self.reward()
         #print("Reward: ", reward)
        
        info = {
              'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
              'episode_service_blocking_rate': (
                                                        self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
              'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
              'episode_bit_rate_blocking_rate': (
                                                          self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / self.episode_bit_rate_requested, 
              'failure':(self.failure_counter)/self.services_processed, 
              'episode_failure':(self.episode_failure_counter/self.episode_services_processed,),
              'failure_slots': (self.failure_counter-self.failure_disjointness)/self.services_processed, #Of total services
              'episode_failure_slots': (self.episode_failure_counter - self.episode_failure_disjointness)/self.episode_services_processed, 
              'failure_disjointness': (self.failure_disjointness)/self.services_processed,
              'episode_failure_disjointness': (self.episode_failure_disjointness)/self.episode_services_processed,
              'length_comparison': self.length_comparison,
              'length_wp': self.length_wp,
              'length_bp': self.length_bp,
              'hops_wp': self.hops_wp,
              'hops_bp': self.hops_bp,
              'number_slots_wp': self.working_slots_2,
              'number_slots_bp': self.backup_slots_2,
              'initial_slots_wp': self.initial_slot_working_2,
              'initial_slots_bp': self.initial_slot_backup_2,
              'average_external_fragmentation_wp': self.curr_ext_fragmentation_wp,
              'average_external_fragmentation_bp': self.curr_ext_fragmentation_bp,
        }
    
        self._new_service = False
        self._next_service()
        return self.observation(), reward, self.episode_services_processed == self.episode_length, info


    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        
        self.episode_failure_disjointness = 0
        self.episode_failure_counter = 0 
        
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                                self.num_spectrum_resources + self.reject_action),
                                               dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + self.reject_action,
                                               self.num_spectrum_resources + self.reject_action),
                                              dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                         dtype=int)

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['external_fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    # We need to add an input: backup path
    def _provision_path(self, working_path: Path, initial_slot_working, number_slots_working,
                        backup_path: Path, initial_slot_backup, number_slots_backup):
        # usage for both paths
        if not self.is_path_free(working_path, initial_slot_working, number_slots_working):
            raise ValueError(
                "Working path {} has not enough capacity on slots {}-{}".format(working_path.node_list, working_path,
                                                                                initial_slot_working,
                                                                                initial_slot_working + number_slots_working))
        if not self.is_path_free(backup_path, initial_slot_backup, number_slots_backup):
            raise ValueError(
                "Backup path {} has not enough capacity on slots {}-{}".format(backup_path.node_list, backup_path,
                                                                               initial_slot_backup,
                                                                               initial_slot_backup + number_slots_backup))

        self.logger.debug(
            '{} assigning working path {} on initial slot {} for {} slots and backup path {} on initial slot {} for {} slots' \
            .format(self.service.service_id, working_path.node_list, initial_slot_working, number_slots_working,
                    backup_path.node_list, initial_slot_backup, number_slots_backup))

        for i in range(len(working_path.node_list) - 1):
            self.topology.graph['available_slots'][
            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]]['index'],
            initial_slot_working:initial_slot_working + number_slots_working] = 0
            self.spectrum_slots_allocation[
            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]]['index'],
            initial_slot_working:initial_slot_working + number_slots_working] = self.service.service_id
            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]]['services'].append(
                self.service)  # Do we need to do that twice?
            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]]['running_services'].append(
                self.service)
            
            self._update_link_stats(working_path.node_list[i], working_path.node_list[i + 1])
            if i == 0:
                self.curr_ext_fragmentation_wp = 0
            self.curr_ext_fragmentation_wp = self.curr_ext_fragmentation_wp + self.tmp

        if len(working_path.node_list) - 1 != 0:
            self.curr_ext_fragmentation_wp = self.curr_ext_fragmentation_wp/(len(working_path.node_list) - 1) 

        for i in range(len(backup_path.node_list) - 1):
            self.topology.graph['available_slots'][ self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]]['index'], initial_slot_backup:initial_slot_backup + number_slots_backup] = 0
            self.spectrum_slots_allocation[ self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]]['index'], initial_slot_backup:initial_slot_backup + number_slots_backup] = self.service.service_id 
            
            self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]]['services'].append(self.service)
            self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]]['running_services'].append(self.service)
            
            
            self._update_link_stats(backup_path.node_list[i], backup_path.node_list[i + 1])
            if i == 0:
                self.curr_ext_fragmentation_bp = 0
            
            self.curr_ext_fragmentation_bp = self.curr_ext_fragmentation_bp + self.tmp

        if len(backup_path.node_list) - 1 != 0:
            self.curr_ext_fragmentation_bp = self.curr_ext_fragmentation_bp/(len(backup_path.node_list) - 1) 

        self.topology.graph['running_services'].append(self.service)
        self.service.route = working_path
        self.service.initial_slot = initial_slot_working

        self.service.backup_route = backup_path
        self.service.initial_slot_backup = initial_slot_backup

        self.service.number_slots = number_slots_working
        self.service.number_slots_backup = number_slots_backup

        self._update_network_stats()
        self.services_accepted += 1
        self.episode_services_accepted += 1

        self.bit_rate_provisioned += self.service.bit_rate
        self.episode_bit_rate_provisioned += self.service.bit_rate

    def _release_path(self, service: Service):  # Doubled to take into account backup
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_slots'][
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
            service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.spectrum_slots_allocation[
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
            service.initial_slot:service.initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(
                service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        # New loop that considers "backup" instead of "working"
        for i in range(len(service.backup_route.node_list) - 1):
            self.topology.graph['available_slots'][
            self.topology[service.backup_route.node_list[i]][service.backup_route.node_list[i + 1]]['index'],
            service.initial_slot_backup:service.initial_slot_backup + service.number_slots_backup] = 1
            self.spectrum_slots_allocation[
            self.topology[service.backup_route.node_list[i]][service.backup_route.node_list[i + 1]]['index'],
            service.initial_slot_backup:service.initial_slot_backup + service.number_slots_backup] = -1
            self.topology[service.backup_route.node_list[i]][service.backup_route.node_list[i + 1]][
                'running_services'].remove(service)
            self._update_link_stats(service.backup_route.node_list[i], service.backup_route.node_list[i + 1])

        self.topology.graph['running_services'].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']
            last_compactness = self.topology.graph['compactness']

            cur_throughput = 0.

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = ((last_throughput * last_update) + (cur_throughput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = throughput

            compactness = ((last_compactness * last_update) + (self._get_network_compactness() * time_diff)) / \
                          self.current_time
            self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :])) / \
                       self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_external_fragmentation = 0.
            cur_link_compactness = 0.
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSADPPEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSADPPEnv.rle(
                        slot_allocation[lambda_min:lambda_max])
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = ((lambda_max - lambda_min) / np.sum(1 - slot_allocation)) * (
                                1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.
                else:
                    cur_link_compactness = 1.

            external_fragmentation = ((last_external_fragmentation * last_update) + (
                        cur_external_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation

            self.tmp = external_fragmentation

            link_compactness = ((last_compactness * last_update) + (
                        cur_link_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        bit_rate = self.rng.randint(self.bit_rate_lower_bound, self.bit_rate_higher_bound)

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

        self.service = Service(self.episode_services_processed, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)
        self._new_service = True

    def _get_path_slot_id(self, action: int) -> (int, int):  # May need to be changed
        """
        Decodes the single action index into the path index and the slot index to be used.

        :param action: the single action index
        :return: path index and initial slot index encoded in the action
        """
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    # def _get_path_block_id(self, action: int) -> (int, int, int, int):  
    #   working_path = action // (self.j*self.j)

    #   block_working = action % self.j

    #   backup_path = (action%self.j) // self.j

    #   block_backup = action % self.j

    #   return working_path, block_working, backup_path, block_backup

    def get_number_slots(self, path: Path) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        return math.ceil(self.service.bit_rate / path.best_modulation['capacity']) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph['available_slots'][
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      initial_slot:initial_slot + number_slots] == 0):
                return False
        return True

    # Function to check if two paths are node disjoint:
    def is_disjoint(self, working_path: Path, backup_path: Path) -> bool:  # takes working and backup as parameters
        if working_path.node_list != backup_path.node_list: 
            for i in range(1, len(working_path.node_list) - 1):  # for each value of the working_path node list,
                if working_path.node_list[
                    i] in backup_path.node_list:  # it checks if the node exists in backup path node list
                    return False  # If it is found
            return True
        else: 
            return False

    def get_available_slots(self, path: Path):
        available_slots = functools.reduce(np.multiply,
                                           self.topology.graph["available_slots"][
                                           [self.topology[path.node_list[i]][path.node_list[i + 1]]['id']
                                            for i in range(len(path.node_list) - 1)], :])
        return available_slots

    def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path):  # Do we need to double that or use it two times ? (my thought: use it twice)
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path])

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path])  # Slots may need to be doubled

        # getting the blocks
        initial_indices, values, lengths = RMSADPPEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[:self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        # Take into account backup paths ?
        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.route.hops
            # sum_slots_paths += (service.number_slots * service.route.hops +
            #  service.number_slots_backup * service.backup_route.hops)

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = \
                RMSADPPEnv.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSADPPEnv.rle(
                    self.topology.graph['available_slots'][self.topology[n1][n2]['index'], lambda_min:lambda_max])
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (self.topology.number_of_edges() /
                                                                           sum_unused_spectrum_blocks)
        else:
            cur_spectrum_compactness = 1.

        return cur_spectrum_compactness


def shortest_path_first_fit(env: RMSADPPEnv) -> int:
    num_slots = env.get_number_slots(env.k_shortest_paths[env.service.source, env.service.destination][0])
    for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
        if env.is_path_free(env.k_shortest_paths[env.service.source, env.service.destination][0], initial_slot,
                            num_slots):
            return [0, initial_slot]
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]


def shortest_available_path_first_fit(env: RMSADPPEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                return [idp, initial_slot]
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]


def least_loaded_path_first_fit(env: RMSADPPEnv) -> int:
    max_free_slots = 0
    action = [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = [idp, initial_slot]
                    max_free_slots = free_slots
                break  # breaks the loop for the initial slot
    return action


class SimpleMatrixObservation(gym.ObservationWrapper):

    def __init__(self, env: RMSADPPEnv):
        super().__init__(env)
        shape = self.env.topology.number_of_nodes() * 2 \
                + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(self.env.service.source_id, self.env.service.destination_id)
        max_node = max(self.env.service.source_id, self.env.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate((source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1) \
            .reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):

    def __init__(self, env: RMSADPPEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(self.env.k_shortest_paths[self.env.service.source,
                                                                            self.env.service.destination][action])
            for initial_slot in range(0, self.env.topology.graph['num_spectrum_resources'] - num_slots):
                if self.env.is_path_free(self.env.k_shortest_paths[self.env.service.source,
                                                                   self.env.service.destination][action],
                                         initial_slot, num_slots):
                    return [action, initial_slot]
        return [self.env.topology.graph['k_paths'], self.env.topology.graph['num_spectrum_resources']]

    def step(self, action):
        return self.env.step(self.action(action))
