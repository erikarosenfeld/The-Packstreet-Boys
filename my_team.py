# the_packstreet_boys.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

import random
import util
import heapq
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


def create_team(first_index, second_index, is_red,
                first='TPBAStarOffensiveAgent', second='TPBDefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class TPBAStarOffensiveAgent(CaptureAgent):
    """
    Advanced offensive agent using A* pathfinding with intelligent mode switching
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode = 'collect'
        self.safe_boundary_positions = []
        self.last_positions = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.compute_boundary_positions(game_state)

    def compute_boundary_positions(self, game_state):
        layout = game_state.data.layout
        boundary_x = layout.width // 2
        if self.red:
            boundary_x -= 1
        
        self.safe_boundary_positions = []
        for y in range(layout.height):
            if not game_state.has_wall(boundary_x, y):
                self.safe_boundary_positions.append((boundary_x, y))

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # Track if we're stuck
        self.last_positions.append(my_pos)
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)

        # Update mode based on situation
        self.update_mode(game_state)
        
        # Get legal actions
        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)

        # Choose action based on current mode
        if self.mode == 'escape':
            return self.escape_action(game_state, actions)
        elif self.mode == 'return':
            return self.return_home_action(game_state, actions)
        elif self.mode == 'hunt_capsule':
            return self.hunt_capsule_action(game_state, actions)
        else:
            return self.collect_food_action(game_state, actions)

    def update_mode(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        food_carrying = my_state.num_carrying
        
        # Get enemy information
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        # Danger detection
        in_danger = False
        if ghosts:
            dangerous_ghosts = [g for g in ghosts if g.scared_timer < 5]
            if dangerous_ghosts:
                dangerous_distances = [self.get_maze_distance(my_pos, g.get_position()) 
                                      for g in dangerous_ghosts]
                min_danger_dist = min(dangerous_distances)
                if min_danger_dist <= 4:
                    in_danger = True

        # Mode decision logic
        if in_danger:
            self.mode = 'escape'
        elif food_carrying >= 2:
            self.mode = 'return'
        else:
            self.mode = 'collect'

    def escape_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        
        # Get ghost positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        # Find safest escape route
        safe_options = []
        for safe_pos in self.safe_boundary_positions:
            dist_to_safe = self.get_maze_distance(my_pos, safe_pos)
            if ghosts:
                ghost_dists_to_safe = [self.get_maze_distance(safe_pos, g.get_position()) for g in ghosts]
                min_ghost_dist_to_safe = min(ghost_dists_to_safe)
            else:
                min_ghost_dist_to_safe = 10
            safety_score = min_ghost_dist_to_safe * 2 - dist_to_safe
            safe_options.append((safety_score, safe_pos))
        
        # Try safest escape routes
        safe_options.sort(reverse=True, key=lambda x: x[0])
        for score, safe_pos in safe_options[:2]:
            best_action = self.a_star_search(game_state, safe_pos, avoid_ghosts=True)
            if best_action:
                return best_action
        
        return self.move_away_from_ghosts(game_state, actions)

    def move_away_from_ghosts(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        if not ghosts:
            return random.choice(actions)
        
        best_action = None
        best_min_distance = -1
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            min_ghost_dist = min([self.get_maze_distance(new_pos, g.get_position()) for g in ghosts])
            dist_to_home = min([self.get_maze_distance(new_pos, safe_pos) for safe_pos in self.safe_boundary_positions])
            score = min_ghost_dist * 3 - dist_to_home
            
            if score > best_min_distance:
                best_min_distance = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)

    def return_home_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        closest_home = min(self.safe_boundary_positions,
                          key=lambda pos: self.get_maze_distance(my_pos, pos))
        
        best_action = self.a_star_search(game_state, closest_home, avoid_ghosts=True)
        if best_action:
            return best_action
        
        return self.get_best_action_by_evaluation(game_state, actions, 'return')

    def hunt_capsule_action(self, game_state, actions):
        capsules = self.get_capsules(game_state)
        if not capsules:
            self.mode = 'collect'
            return self.collect_food_action(game_state, actions)
        
        my_pos = game_state.get_agent_position(self.index)
        closest_capsule = min(capsules, key=lambda cap: self.get_maze_distance(my_pos, cap))
        best_action = self.a_star_search(game_state, closest_capsule, avoid_ghosts=False)
        
        if best_action:
            return best_action
        
        return self.get_best_action_by_evaluation(game_state, actions, 'capsule')

    def collect_food_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            return random.choice(actions)
        
        target = self.select_best_food_target(game_state, food_list)
        best_action = self.a_star_search(game_state, target, avoid_ghosts=True)
        
        if best_action:
            return best_action
        
        return self.get_best_action_by_evaluation(game_state, actions, 'collect')

    def select_best_food_target(self, game_state, food_list):
        my_pos = game_state.get_agent_position(self.index)
        
        # Get ghost positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_positions = [a.get_position() for a in enemies 
                          if not a.is_pacman and a.get_position() is not None 
                          and a.scared_timer < 2]
        
        best_food = None
        best_score = float('-inf')
        
        for food in food_list:
            food_dist = self.get_maze_distance(my_pos, food)
            home_dist = min([self.get_maze_distance(food, safe_pos) for safe_pos in self.safe_boundary_positions])
            
            ghost_safety = float('inf')
            if ghost_positions:
                ghost_safety = min([self.get_maze_distance(food, ghost_pos) for ghost_pos in ghost_positions])
            
            if ghost_safety < 3:
                score = -1000
            elif ghost_safety < 5:
                score = -food_dist * 2 + ghost_safety * 5 - home_dist * 0.3
            else:
                score = -food_dist + ghost_safety * 0.5 - home_dist * 0.2
            
            if (not self.red and food[0] < my_pos[0]) or (self.red and food[0] > my_pos[0]):
                score += 2
            
            if score > best_score:
                best_score = score
                best_food = food
        
        if best_score == -1000:
            if ghost_positions:
                best_food = max(food_list, 
                              key=lambda f: min([self.get_maze_distance(f, gp) for gp in ghost_positions]))
            else:
                best_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        return best_food if best_food else food_list[0]

    def a_star_search(self, game_state, goal, avoid_ghosts=True):
        start_time = time.time()
        max_time = 0.8
        
        start_pos = game_state.get_agent_position(self.index)
        
        if start_pos == goal:
            return None
        
        # Get ghost danger zones
        ghost_danger_zones = set()
        if avoid_ghosts:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            for enemy in enemies:
                if not enemy.is_pacman and enemy.get_position() is not None:
                    if enemy.scared_timer < 2:
                        ghost_pos = enemy.get_position()
                        ghost_danger_zones.add(ghost_pos)
                        layout = game_state.data.layout
                        x, y = int(ghost_pos[0]), int(ghost_pos[1])
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if abs(dx) + abs(dy) <= 2:
                                    danger_pos = (x + dx, y + dy)
                                    if (0 <= danger_pos[0] < layout.width and 
                                        0 <= danger_pos[1] < layout.height and
                                        not game_state.has_wall(int(danger_pos[0]), int(danger_pos[1]))):
                                        ghost_danger_zones.add(danger_pos)
        
        # A* implementation
        frontier = []
        heapq.heappush(frontier, (0, start_pos, [], 0))
        explored = set()
        
        max_iterations = 300
        iterations = 0
        
        while frontier and iterations < max_iterations:
            if time.time() - start_time > max_time:
                return self.get_fallback_action(game_state, goal)
                
            iterations += 1
            _, current_pos, path, cost = heapq.heappop(frontier)
            
            if current_pos == goal:
                if path:
                    return path[0]
                return None
            
            if current_pos in explored:
                continue
            
            explored.add(current_pos)
            
            # Get valid neighboring positions
            x, y = current_pos
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            
            for next_pos in neighbors:
                if next_pos in explored:
                    continue
                
                layout = game_state.data.layout
                # Check if position is valid
                if (0 <= next_pos[0] < layout.width and 
                    0 <= next_pos[1] < layout.height and
                    not game_state.has_wall(int(next_pos[0]), int(next_pos[1]))):
                    
                    # Calculate base cost
                    new_cost = cost + 1
                    
                    # Add ghost penalties if needed
                    if avoid_ghosts and next_pos in ghost_danger_zones:
                        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                        ghost_positions = [a.get_position() for a in enemies 
                                         if not a.is_pacman and a.get_position() is not None 
                                         and a.scared_timer < 2]
                        if ghost_positions:
                            min_ghost_dist = min([self.get_maze_distance(next_pos, gp) for gp in ghost_positions])
                            if min_ghost_dist <= 2:
                                new_cost += 50 * (3 - min_ghost_dist)  # Penalty based on distance
                    
                    # Calculate priority
                    heuristic = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                    priority = new_cost + heuristic
                    
                    # Determine action
                    if not path:
                        dx = next_pos[0] - current_pos[0]
                        dy = next_pos[1] - current_pos[1]
                        if dx > 0:
                            action = Directions.EAST
                        elif dx < 0:
                            action = Directions.WEST
                        elif dy > 0:
                            action = Directions.NORTH
                        else:
                            action = Directions.SOUTH
                        new_path = [action]
                    else:
                        new_path = path
                    
                    heapq.heappush(frontier, (priority, next_pos, new_path, new_cost))
        
        return self.get_fallback_action(game_state, goal)

    def get_fallback_action(self, game_state, goal):
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)
        
        best_action = None
        best_dist = float('inf')
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, goal)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else random.choice(actions)

    def get_best_action_by_evaluation(self, game_state, actions, mode):
        values = []
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            value = self.evaluate_state(successor, mode)
            values.append(value)
        
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        return random.choice(best_actions)

    def evaluate_state(self, game_state, mode):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        score = 0
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            score += min_ghost_dist * 10
        
        if mode == 'return':
            boundary_dist = min([self.get_maze_distance(my_pos, pos) for pos in self.safe_boundary_positions])
            score -= boundary_dist * 100
        elif mode == 'collect':
            food_list = self.get_food(game_state).as_list()
            if food_list:
                min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
                score -= min_food_dist * 5
        elif mode == 'capsule':
            capsules = self.get_capsules(game_state)
            if capsules:
                min_cap_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
                score -= min_cap_dist * 20
        
        score += self.get_score(game_state) * 100
        
        return score


class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class TPBDefensiveReflexAgent(ReflexCaptureAgent):
    #We updated the defensive reflex agent in order to make it more efficient.
    #Its main aims are to guard our food and to eat enemy pacmans approaching
    #to our side.

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        #determines whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        #measure distance of invaders that can be seen 
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            #in case there are no visible invaders: we patrol defensively close to our food
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                #move towards the center of our food to patrol it
                min_dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['food_defense_distance'] = min_dist_to_food

        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -25, 'stop': -100, 'reverse': -2, 'food_defense_distance': -3}
