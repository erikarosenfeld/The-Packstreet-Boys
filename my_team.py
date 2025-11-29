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


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AStarOffensiveAgent', second='TPBDefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class AStarOffensiveAgent(CaptureAgent):
    """
    Advanced offensive agent using A* pathfinding, behavior trees,
    and intelligent decision making
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.target_food = None
        self.mode = 'collect'  # Modes: 'collect', 'return', 'escape', 'hunt_capsule'
        self.safe_boundary_positions = []
        self.food_collected_this_trip = 0
        self.last_positions = []
        self.stuck_counter = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # Pre-compute safe boundary positions for quick returns
        self.compute_boundary_positions(game_state)

    def compute_boundary_positions(self, game_state):
        """
        Pre-compute all valid positions on our side of the boundary
        """
        layout = game_state.data.layout
        boundary_x = layout.width // 2
        
        if self.red:
            boundary_x -= 1
        
        self.safe_boundary_positions = []
        for y in range(layout.height):
            if not game_state.has_wall(boundary_x, y):
                self.safe_boundary_positions.append((boundary_x, y))

    def choose_action(self, game_state):
        """
        Main decision-making function with mode-based behavior
        """
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Track if we're stuck
        self.last_positions.append(my_pos)
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)
            if len(set(self.last_positions)) <= 2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        # Update mode based on situation
        old_mode = self.mode
        self.update_mode(game_state)
        
        # DEBUG: Print mode changes
        if old_mode != self.mode:
            food_carrying = my_state.num_carrying
            print(f"DEBUG: Mode changed from {old_mode} to {self.mode} (carrying {food_carrying} food)")
        
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
        else:  # collect mode
            return self.collect_food_action(game_state, actions)

    def update_mode(self, game_state):
        """
        SIMPLIFIED mode switching with debug prints
        """
        my_state = game_state.get_agent_state(self.index)
        food_carrying = my_state.num_carrying
        
        # DEBUG: Print when we have food but aren't returning
        if food_carrying >= 2 and self.mode != 'return':
            print(f"DEBUG: Carrying {food_carrying} food but mode is {self.mode}")
        
        # SUPER SIMPLE: Return after 2 food, otherwise collect
        if food_carrying >= 2:
            self.mode = 'return'
            print(f"FORCING RETURN: Carrying {food_carrying} food")
        else:
            self.mode = 'collect'

    def escape_action(self, game_state, actions):
        """
        SIMPLIFIED but more effective escape
        """
        my_pos = game_state.get_agent_position(self.index)
        print("DEBUG: ESCAPE MODE ACTIVATED")
        
        # Strategy: Just find the closest safe spot and go there
        closest_safe_pos = min(self.safe_boundary_positions,
                              key=lambda pos: self.get_maze_distance(my_pos, pos))
        
        # Use simple pathfinding without complex ghost avoidance
        best_action = self.get_fallback_action(game_state, closest_safe_pos)
        
        # If that doesn't work, use the move that maximizes distance from ghosts
        if not best_action:
            best_action = self.move_away_from_ghosts(game_state, actions)
        
        return best_action if best_action else random.choice(actions)

    def move_away_from_ghosts(self, game_state, actions):
        """
        SIMPLIFIED ghost avoidance - just maximize minimum distance
        """
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
            
            # Calculate minimum distance to any ghost
            min_ghost_dist = min([self.get_maze_distance(new_pos, g.get_position()) 
                                 for g in ghosts])
            
            # Also check if this gets us closer to home
            dist_to_home = min([self.get_maze_distance(new_pos, safe_pos) 
                               for safe_pos in self.safe_boundary_positions])
            
            # Combined score: prioritize ghost distance, then home distance
            score = min_ghost_dist * 3 - dist_to_home
            
            if score > best_min_distance:
                best_min_distance = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)

    def return_home_action(self, game_state, actions):
        """
        Return home efficiently using A* with debug prints
        """
        my_pos = game_state.get_agent_position(self.index)
        food_carrying = game_state.get_agent_state(self.index).num_carrying
        
        print(f"DEBUG: Trying to return home with {food_carrying} food from position {my_pos}")
        
        # Find closest boundary position
        closest_home = min(self.safe_boundary_positions,
                          key=lambda pos: self.get_maze_distance(my_pos, pos))
        
        print(f"DEBUG: Closest home position is {closest_home}")
        
        best_action = self.a_star_search(game_state, closest_home, avoid_ghosts=True)
        
        if best_action:
            print(f"DEBUG: A* found path home: {best_action}")
            return best_action
        else:
            print("DEBUG: A* failed to find path home, using fallback")
        
        # Fallback
        return self.get_best_action_by_evaluation(game_state, actions, 'return')

    def hunt_capsule_action(self, game_state, actions):
        """
        Go for power capsule strategically
        """
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
        """
        Intelligent food collection with strategic target selection
        """
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            return random.choice(actions)
        
        # Select best food target
        target = self.select_best_food_target(game_state, food_list)
        
        # Use A* to reach target
        best_action = self.a_star_search(game_state, target, avoid_ghosts=True)
        
        if best_action:
            return best_action
        
        # Fallback to evaluation-based selection
        return self.get_best_action_by_evaluation(game_state, actions, 'collect')

    def select_best_food_target(self, game_state, food_list):
        """
        Select the best food pellet with ENHANCED safety consideration
        """
        my_pos = game_state.get_agent_position(self.index)
        
        # Get ghost positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_positions = [a.get_position() for a in enemies 
                          if not a.is_pacman and a.get_position() is not None 
                          and a.scared_timer < 2]
        
        best_food = None
        best_score = float('-inf')
        
        for food in food_list:
            # Calculate distance to food
            food_dist = self.get_maze_distance(my_pos, food)
            
            # Calculate distance back home from food
            home_dist = min([self.get_maze_distance(food, safe_pos) 
                           for safe_pos in self.safe_boundary_positions])
            
            # Calculate minimum distance to any ghost - CRITICAL FACTOR
            ghost_safety = float('inf')
            if ghost_positions:
                ghost_safety = min([self.get_maze_distance(food, ghost_pos) 
                                  for ghost_pos in ghost_positions])
            
            # ENHANCED: Don't go for food that's too close to ghosts
            if ghost_safety < 3:
                # Very dangerous food, heavily penalize
                score = -1000
            elif ghost_safety < 5:
                # Somewhat dangerous
                score = -food_dist * 2 + ghost_safety * 5 - home_dist * 0.3
            else:
                # Safe food, normal scoring
                score = -food_dist + ghost_safety * 0.5 - home_dist * 0.2
            
            # Bonus for food that's on the way home
            if (not self.red and food[0] < my_pos[0]) or (self.red and food[0] > my_pos[0]):
                score += 2  # Food is towards home
            
            if score > best_score:
                best_score = score
                best_food = food
        
        # If all food is dangerous, pick the safest one
        if best_score == -1000:
            # Just pick the one farthest from ghosts
            if ghost_positions:
                best_food = max(food_list, 
                              key=lambda f: min([self.get_maze_distance(f, gp) 
                                               for gp in ghost_positions]))
            else:
                best_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        return best_food if best_food else food_list[0]

    def a_star_search(self, game_state, goal, avoid_ghosts=True):
        """
        ENHANCED A* pathfinding with better ghost avoidance and timeout
        Returns the first action to take
        """
        start_time = time.time()
        max_time = 0.8  # 800ms timeout
        
        start_pos = game_state.get_agent_position(self.index)
        
        if start_pos == goal:
            return None
        
        # Get ghost positions and their predicted next positions
        ghost_danger_zones = set()
        if avoid_ghosts:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            for enemy in enemies:
                if not enemy.is_pacman and enemy.get_position() is not None:
                    if enemy.scared_timer < 2:
                        ghost_pos = enemy.get_position()
                        ghost_danger_zones.add(ghost_pos)
                        
                        # Add all positions within 2 steps of ghost as danger zones
                        x, y = int(ghost_pos[0]), int(ghost_pos[1])
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if abs(dx) + abs(dy) <= 2:  # Manhattan distance <= 2
                                    danger_pos = (x + dx, y + dy)
                                    if not game_state.has_wall(int(danger_pos[0]), int(danger_pos[1])):
                                        ghost_danger_zones.add(danger_pos)
        
        # A* implementation with enhanced ghost avoidance
        frontier = []
        heapq.heappush(frontier, (0, start_pos, [], 0))
        explored = set()
        
        max_iterations = 300  # Prevent infinite loops
        iterations = 0
        
        while frontier and iterations < max_iterations:
            # Check timeout
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
                
                # Check if position is valid
                if game_state.has_wall(int(next_pos[0]), int(next_pos[1])):
                    continue
                
                # Calculate cost with HEAVY ghost penalties
                new_cost = cost + 1
                
                # CRITICAL: If in ghost danger zone, add massive penalty
                if next_pos in ghost_danger_zones:
                    # Check exact distance to nearest ghost
                    if avoid_ghosts:
                        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                        ghost_positions = [a.get_position() for a in enemies 
                                         if not a.is_pacman and a.get_position() is not None 
                                         and a.scared_timer < 2]
                        if ghost_positions:
                            min_ghost_dist = min([self.get_maze_distance(next_pos, gp) 
                                                 for gp in ghost_positions])
                            
                            if min_ghost_dist == 0:
                                new_cost += 200  # On top of ghost
                            elif min_ghost_dist == 1:
                                new_cost += 100  # Next to ghost
                            elif min_ghost_dist == 2:
                                new_cost += 50   # Near ghost
                
                # Heuristic: Manhattan distance to goal
                heuristic = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                priority = new_cost + heuristic
                
                # Determine action that led to next_pos
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
        
        return self.get_fallback_action(game_state, goal)  # Fallback if no path found

    def get_fallback_action(self, game_state, goal):
        """Simple greedy fallback when A* times out or fails"""
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
        """
        Fallback evaluation function
        """
        values = []
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            value = self.evaluate_state(successor, mode)
            values.append(value)
        
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        return random.choice(best_actions)

    def evaluate_state(self, game_state, mode):
        """
        Evaluate game state based on current mode
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        score = 0
        
        # Get enemy information
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        # Ghost distance factor
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            score += min_ghost_dist * 10
        
        # Mode-specific evaluation
        if mode == 'return':
            # Reward getting closer to home
            boundary_dist = min([self.get_maze_distance(my_pos, pos) 
                               for pos in self.safe_boundary_positions])
            score -= boundary_dist * 100
        
        elif mode == 'collect':
            # Reward proximity to food
            food_list = self.get_food(game_state).as_list()
            if food_list:
                min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
                score -= min_food_dist * 5
        
        elif mode == 'capsule':
            # Reward proximity to capsules
            capsules = self.get_capsules(game_state)
            if capsules:
                min_cap_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
                score -= min_cap_dist * 20
        
        # General score improvement
        score += self.get_score(game_state) * 100
        
        return score


# Reflex Agent Base Class (needed for defensive agent)
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
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
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class TPBDefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
