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
    More aggressive offensive agent using A* pathfinding
    Improvements:
    - FLANKING: Moves to other side of board if blocked at border
    - SMART CAPSULES: Hunts power pellets when threatened
    - AGGRESSIVE EATING: Uses power pellet time to eat food, not ghosts
    - ANTI-ROAMING: Falls back to nearest food if advanced targeting fails
    - ANTI-LOOPING: Smarter history to break oscillating loops
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode = 'collect'
        self.safe_boundary_positions = []
        self.last_positions = []
        self.stuck_counter = 0
        self.red_boundary_x = 0
        self.blue_boundary_x = 0
    
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.compute_boundary_positions(game_state)
    
    def compute_boundary_positions(self, game_state):
        layout = game_state.data.layout
        self.red_boundary_x = (layout.width // 2) - 1
        self.blue_boundary_x = layout.width // 2
        
        target_x = self.red_boundary_x if self.red else self.blue_boundary_x
        
        self.safe_boundary_positions = []
        for y in range(layout.height):
            if not game_state.has_wall(target_x, y):
                self.safe_boundary_positions.append((target_x, y))
    
    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # --- IMPROVED HISTORY LOGIC ---
        # Keep track of last 20 positions to catch larger loops
        self.last_positions.append(my_pos)
        if len(self.last_positions) > 20:
            self.last_positions.pop(0)
        
        # Stuck Detection: Have we been here > 3 times recently?
        if self.last_positions.count(my_pos) > 3:
            self.stuck_counter = 4  # Force breakout behavior for next 4 moves
        
        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)

        # If stuck, use special "Breakout" logic
        if self.stuck_counter > 0:
            self.stuck_counter -= 1
            return self.get_stuck_breakout_action(game_state, actions)
        # -------------------------------
        
        # Update mode based on situation
        self.update_mode(game_state)
        
        # Choose action based on current mode
        if self.mode == 'escape':
            return self.escape_action(game_state, actions)
        elif self.mode == 'return':
            return self.return_home_action(game_state, actions)
        elif self.mode == 'hunt_capsule':
            return self.hunt_capsule_action(game_state, actions)
        else:
            return self.collect_food_action(game_state, actions)

    def get_stuck_breakout_action(self, game_state, actions):
        """
        When stuck, choose the action that leads to the neighbor we have visited 
        the LEAST in our recent history. This forces exploration of new tiles.
        """
        # 1. Filter out actions that lead immediately to death (ghosts)
        safe_actions = []
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        # Treat ghosts as dangerous for breakout logic unless firmly scared
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 3]

        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_position(self.index)
            
            is_safe = True
            for ghost in ghosts:
                if self.get_maze_distance(next_pos, ghost.get_position()) <= 1:
                    is_safe = False
                    break
            if is_safe:
                safe_actions.append(action)
        
        if not safe_actions:
            return random.choice(actions)

        # 2. From safe actions, pick the one leading to the least visited position
        best_action = None
        min_visits = float('inf')
        
        for action in safe_actions:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_position(self.index)
            visits = self.last_positions.count(next_pos)
            
            if visits < min_visits:
                min_visits = visits
                best_action = action
            elif visits == min_visits and random.random() < 0.5:
                 best_action = action
                
        return best_action if best_action else random.choice(safe_actions)
    
    def update_mode(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        food_carrying = my_state.num_carrying
        
        # Get enemy information
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        # Determine danger status
        # We are only in danger if there is a ghost that is NOT scared (or timer is low)
        dangerous_ghosts = [g for g in ghosts if g.scared_timer <= 5]
        
        in_danger = False
        danger_dist = float('inf')
        
        if dangerous_ghosts:
            dangerous_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in dangerous_ghosts]
            danger_dist = min(dangerous_distances)
            if danger_dist <= 3:
                in_danger = True
        
        # Check for capsules
        capsules = self.get_capsules(game_state)
        closest_capsule_dist = float('inf')
        if capsules:
            closest_capsule_dist = min([self.get_maze_distance(my_pos, c) for c in capsules])

        # Distance to home
        dist_to_home = min([self.get_maze_distance(my_pos, pos) for pos in self.safe_boundary_positions])
        
        # --- NEW STRATEGY LOGIC ---
        
        # 1. SCARED / FREE EATING MODE
        # If there are ghosts but they are all scared (and we are not in danger from a non-scared one),
        # we should eat as much as possible.
        scared_ghosts_exist = any(g.scared_timer > 5 for g in ghosts)
        
        if scared_ghosts_exist and not in_danger:
            # We are safe for now. Check if we need to run home before timer expires.
            # Find minimum scared timer
            timers = [g.scared_timer for g in ghosts if g.scared_timer > 0]
            min_timer = min(timers) if timers else 0
            
            # If timer is running out (it takes time to get home), return now
            if min_timer < dist_to_home + 5:
                 self.mode = 'return'
            else:
                 self.mode = 'collect' # Eat everything!
            return

        # 2. STANDARD LOGIC (Danger or No Ghosts)
        
        # Dynamic banking threshold
        if in_danger:
            banking_threshold = 1 if food_carrying > 0 else 0
        elif danger_dist <= 4:
            banking_threshold = 4
        else:
            banking_threshold = 6 # Standard carrying limit
        
        # Factor in distance - if far from home, bank earlier
        if dist_to_home > 12:
            banking_threshold = min(banking_threshold, 4)
        
        if in_danger:
            # Fight or Flight
            if capsules and closest_capsule_dist <= 7:
                self.mode = 'hunt_capsule'
            elif food_carrying > 0:
                self.mode = 'escape'
            else:
                # No food, but threatened. Try to get capsule or escape.
                if capsules and closest_capsule_dist <= 10:
                    self.mode = 'hunt_capsule'
                else:
                    self.mode = 'escape'
        elif food_carrying >= banking_threshold:
            self.mode = 'return'
        else:
            self.mode = 'collect'
            # Opportunistic capsule grab
            if capsules and closest_capsule_dist < 6:
                self.mode = 'hunt_capsule'
    
    def escape_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        # Only fear dangerous ghosts
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 5]
        
        safe_options = []
        for safe_pos in self.safe_boundary_positions:
            dist_to_safe = self.get_maze_distance(my_pos, safe_pos)
            if ghosts:
                ghost_dists_to_safe = [self.get_maze_distance(safe_pos, g.get_position()) for g in ghosts]
                min_ghost_dist_to_safe = min(ghost_dists_to_safe)
            else:
                min_ghost_dist_to_safe = 10
            
            safety_score = min_ghost_dist_to_safe * 3 - dist_to_safe
            safe_options.append((safety_score, safe_pos))
        
        safe_options.sort(reverse=True, key=lambda x: x[0])
        
        # Try top 3 safest routes
        for score, safe_pos in safe_options[:3]:
            best_action = self.a_star_search(game_state, safe_pos, avoid_ghosts=True)
            if best_action:
                return best_action
        
        return self.move_away_from_ghosts(game_state, actions)
    
    def move_away_from_ghosts(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 3]
        
        if not ghosts:
            return random.choice(actions)
        
        best_action = None
        best_score = -float('inf')
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            
            min_ghost_dist = min([self.get_maze_distance(new_pos, g.get_position()) for g in ghosts])
            dist_to_home = min([self.get_maze_distance(new_pos, safe_pos) for safe_pos in self.safe_boundary_positions])
            
            score = min_ghost_dist * 4 - dist_to_home
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def return_home_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        
        # Greedy "One More Bite"
        food_list = self.get_food(game_state).as_list()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 3]
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_position(self.index)
            if next_pos in food_list:
                is_safe = True
                if ghosts:
                    dists_to_ghosts = [self.get_maze_distance(next_pos, g.get_position()) for g in ghosts]
                    if dists_to_ghosts and min(dists_to_ghosts) <= 2:
                        is_safe = False
                if is_safe:
                    return action

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
        
        best_action = self.a_star_search(game_state, closest_capsule, avoid_ghosts=True)
        
        if best_action:
            return best_action
        
        return self.get_best_action_by_evaluation(game_state, actions, 'capsule')

    def collect_food_action(self, game_state, actions):
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 5]
        
        # 1. FLANKING CHECK
        boundary_x = self.red_boundary_x if self.red else self.blue_boundary_x
        dist_to_boundary = abs(my_pos[0] - boundary_x)
        
        if dist_to_boundary <= 3 and ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_dist = min(ghost_dists)
            
            if min_ghost_dist < 6:
                flank_target = self.get_flanking_target(game_state, ghosts)
                if flank_target:
                    action = self.a_star_search(game_state, flank_target, avoid_ghosts=True)
                    if action:
                        return action

        # 2. STANDARD COLLECTION
        if not food_list:
            if self.get_score(game_state) > 0:
                self.mode = 'return'
                return self.return_home_action(game_state, actions)
            else:
                return random.choice(actions)
        
        # --- IMPROVED TARGET SELECTION AND ROAMING FIX ---
        target = self.select_best_food_target(game_state, food_list)
        
        # Pass avoid_ghosts=False if ghosts are scared, True otherwise (handled inside A*)
        best_action = self.a_star_search(game_state, target, avoid_ghosts=False)
        
        if best_action:
            return best_action
        
        # 3. FALLBACK: Simple BFS to Nearest Food (Anti-Roaming)
        # If the advanced target is unreachable or A* failed, just go to the closest dot.
        closest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        fallback_action = self.a_star_search(game_state, closest_food, avoid_ghosts=False)
        
        if fallback_action:
            return fallback_action
        
        return self.get_best_action_by_evaluation(game_state, actions, 'collect')

    def get_flanking_target(self, game_state, ghosts):
        ghost_positions = [g.get_position() for g in ghosts]
        if not ghost_positions:
            return None
            
        best_target = None
        max_safety_score = -float('inf')
        
        for boundary_pos in self.safe_boundary_positions:
            min_dist_to_ghost = min([self.get_maze_distance(boundary_pos, gp) for gp in ghost_positions])
            my_pos = game_state.get_agent_position(self.index)
            dist_to_me = self.get_maze_distance(my_pos, boundary_pos)
            
            score = (min_dist_to_ghost * 10) - dist_to_me
            
            if score > max_safety_score:
                max_safety_score = score
                best_target = boundary_pos
                
        return best_target

    def select_best_food_target(self, game_state, food_list):
        my_pos = game_state.get_agent_position(self.index)
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        # Only care about dangerous ghosts for penalty calculation
        dangerous_ghosts = [a.get_position() for a in enemies
                           if not a.is_pacman and a.get_position() is not None
                           and a.scared_timer < 3]
        
        # If no dangerous ghosts, strictly prioritize proximity (greedy eating)
        if not dangerous_ghosts:
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        best_food = None
        best_score = -float('inf')
        
        for food in food_list:
            food_dist = self.get_maze_distance(my_pos, food)
            
            # --- Depth Score (Go deeper) ---
            boundary_x = game_state.data.layout.width // 2
            if self.red:
                enemy_side = food[0] < boundary_x
                depth = (boundary_x - food[0]) if enemy_side else 0
            else:
                enemy_side = food[0] > boundary_x
                depth = (food[0] - boundary_x) if enemy_side else 0
            
            # --- Ghost Penalty ---
            ghost_penalty = 0
            if dangerous_ghosts:
                min_ghost_dist = min([self.get_maze_distance(food, gp) for gp in dangerous_ghosts])
                if min_ghost_dist < 2: ghost_penalty = -100
                elif min_ghost_dist < 4: ghost_penalty = -20
                elif min_ghost_dist < 6: ghost_penalty = -5
            
            home_dist = min([self.get_maze_distance(food, safe_pos) for safe_pos in self.safe_boundary_positions])
            
            # Score:
            # - High penalty for distance (get closest safe food)
            # - Bonus for depth (don't just eat border food)
            # - Penalty for ghosts
            score = -food_dist * 2.0 + depth * 1.5 + ghost_penalty - home_dist * 0.2
            
            if score > best_score:
                best_score = score
                best_food = food
        
        return best_food if best_food else food_list[0]
    
    def is_on_enemy_side(self, pos, boundary_x):
        if self.red:
            return pos[0] < boundary_x
        else:
            return pos[0] > boundary_x
    
    def a_star_search(self, game_state, goal, avoid_ghosts=False):
        start_time = time.time()
        max_time = 0.8
        
        start_pos = game_state.get_agent_position(self.index)
        
        if start_pos == goal:
            return None
        
        ghost_danger_zones = set()
        # Even if avoid_ghosts is True, we only avoid NON-SCARED ghosts.
        # This allows us to walk through scared ghosts to get food.
        if avoid_ghosts:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            for enemy in enemies:
                if not enemy.is_pacman and enemy.get_position() is not None:
                    # ONLY avoid if scared_timer is low
                    if enemy.scared_timer < 3:
                        ghost_pos = enemy.get_position()
                        ghost_danger_zones.add(ghost_pos)
                        layout = game_state.data.layout
                        x, y = int(ghost_pos[0]), int(ghost_pos[1])
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if abs(dx) + abs(dy) <= 1:
                                    danger_pos = (x + dx, y + dy)
                                    if (0 <= danger_pos[0] < layout.width and
                                        0 <= danger_pos[1] < layout.height and
                                        not game_state.has_wall(int(danger_pos[0]), int(danger_pos[1]))):
                                        ghost_danger_zones.add(danger_pos)
        
        frontier = []
        heapq.heappush(frontier, (0, start_pos, [], 0))
        explored = set()
        
        max_iterations = 400
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
            
            x, y = current_pos
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            
            for next_pos in neighbors:
                if next_pos in explored:
                    continue
                
                layout = game_state.data.layout
                if (0 <= next_pos[0] < layout.width and
                    0 <= next_pos[1] < layout.height and
                    not game_state.has_wall(int(next_pos[0]), int(next_pos[1]))):
                    
                    new_cost = cost + 1
                    
                    # Apply penalty only if it's a danger zone (dangerous ghost)
                    if next_pos in ghost_danger_zones:
                         new_cost += 200 # Heavy penalty
                    
                    heuristic = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                    priority = new_cost + heuristic
                    
                    if not path:
                        dx = next_pos[0] - current_pos[0]
                        dy = next_pos[1] - current_pos[1]
                        if dx > 0: action = Directions.EAST
                        elif dx < 0: action = Directions.WEST
                        elif dy > 0: action = Directions.NORTH
                        else: action = Directions.SOUTH
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
            score += min_ghost_dist * 8
        
        if mode == 'return':
            boundary_dist = min([self.get_maze_distance(my_pos, pos) for pos in self.safe_boundary_positions])
            score -= boundary_dist * 100
            score += my_state.num_carrying * 50
        elif mode == 'collect':
            food_list = self.get_food(game_state).as_list()
            if food_list:
                min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
                score -= min_food_dist * 8
                boundary_x = game_state.data.layout.width // 2
                if self.is_on_enemy_side(my_pos, boundary_x):
                    score += 20
        elif mode == 'capsule':
            capsules = self.get_capsules(game_state)
            if capsules:
                min_cap_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
                score -= min_cap_dist * 25
        
        score += self.get_score(game_state) * 200
        
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
    # We updated the defensive reflex agent in order to make it more efficient.
    # Its main aims are to guard our food and to eat enemy pacmans approaching
    # to our side.
    # Fixed food bug (guards our food, not enemy food)
    # Now reacts to eaten food

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Determines whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0
        
        # Measure distance of invaders that can be seen
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_dist = min(dists)
            
            # --- SCARED LOGIC ---
            # If we are scared, we want to RUN AWAY (maximize distance)
            if my_state.scared_timer > 0:
                features['scared_distance'] = min_dist # We will give this a POSITIVE weight
                features['invader_distance'] = 0       # Ignore chasing
            else:
                # If we are not scared, we want to CHASE (minimize distance)
                features['invader_distance'] = min_dist
                features['scared_distance'] = 0
            # -----------------------------
        else:
            # No visible invaders, guard our food
            
            # Use get_food_you_are_defending instead of get_food
            food_defending = self.get_food_you_are_defending(successor).as_list()
            if len(food_defending) > 0:
                min_dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_defending])
                features['food_defense_distance'] = min_dist_to_food
            
            # React to missing food (enemy might be nearby but invisible)
            prev_food = self.get_food_you_are_defending(game_state).as_list()
            curr_food = food_defending
            if len(prev_food) > len(curr_food):
                # Food was just eaten, move toward where it was
                eaten_food = set(prev_food) - set(curr_food)
                if eaten_food:
                    eaten_pos = list(eaten_food)[0]
                    features['eaten_food_distance'] = self.get_maze_distance(my_pos, eaten_pos)
        
        # Penalties for stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        return features
    
    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1500,  # More aggressive against invaders
            'on_defense': 100,
            'invader_distance': -40,  # Chase invaders more aggressively
            'scared_distance': 100,   # Run away when scared (Positive = bigger distance is better)
            'stop': -100,
            'reverse': -2,
            'food_defense_distance': -5,  # Stay closer to our food
            'eaten_food_distance': -50  # Strong reaction to eaten food
        }
