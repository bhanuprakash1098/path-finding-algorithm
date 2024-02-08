#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Artificial Intelligence
Date: 02/03/2024
MP1: Robot navigation
SEMESTER: Spring 2024
NAME: Bhanuprakash Banda
"""

import numpy as np
import queue  # Needed for frontier queue
from heapq import heapify


class MazeState:
    SPACE = 0
    WALL = 1
    EXIT = 2
    VISITED = 3
    PATH = 4
    START_MARK = 5
    END_MARK = 6

    MAZE_FILE = 'maze2024.txt'
    maze = np.loadtxt(MAZE_FILE, dtype=np.int32)
    start = tuple(np.array(np.where(maze == 5)).flatten())
    ends = np.where(maze == 2)
    move_num = 0  # Used by show_path() to count moves in the solution path

    def reset_state(self):
        """
        Reset the maze state to its initial configuration.

        This method reloads the maze from the file, resets the start position,
        exit positions, and move counter.

        Returns:
        - None
        """
        MazeState.maze = np.loadtxt(MazeState.MAZE_FILE, dtype=np.int32)
        MazeState.start = tuple(np.array(np.where(MazeState.maze == 5)).flatten())
        MazeState.ends = np.where(MazeState.maze == 2)
        MazeState.move_num = 0

    def __init__(self, conf=start, g=0, pred_state=None, pred_action=None):
        """
        Initialize the maze state.

        Args:
        - conf: Configuration of the state (default: start)
        - g: Path cost (default: 0)
        - pred_state: Predecessor state (default: None)
        - pred_action: Action from predecessor state to current state (default: None)

        Returns:
        - None
        """
        self.pos = conf  # Configuration of the state - current coordinates
        self.gcost = g  # Path cost
        self.pred = pred_state  # Predecesor state
        self.action_from_pred = pred_action  # Action from predecesor state to current state
        self.hcost = self.heuristic()  # Heuristic

    def __hash__(self):
        """
        Generate a hash value for the state.

        Returns:
        - Hash value based on the position of the state.
        """
        return self.pos.__hash__()

    def is_goal(self):
        """
        Check if the current position is the exit.

        Returns:
        - True if the current position corresponds to the exit.
        """
        return self.maze[self.pos] == MazeState.EXIT

    def __eq__(self, other):
        """
        Check if two states are equal.

        Args:
        - other: Another state to compare with.

        Returns:
        - True if the positions of the two states are equal.
        """
        return self.pos == other.pos

    def __lt__(self, other):
        """
        Compare states based on their total cost.

        This method is required for heapq operations to enable priority queue functionality.

        Args:
        - other: Another state to compare with.

        Returns:
        - True if the total cost of this state is less than the total cost of the other state.
        """
        return (self.gcost + self.hcost) < (other.gcost + other.hcost)

    def __str__(self):
        """
        Return the maze representation of the state.

        Returns:
        - String representation of the maze with start and exit markers.
        """
        a = np.array(self.maze)
        a[self.start] = MazeState.START_MARK
        a[self.ends] = MazeState.EXIT
        return str(a)

    move_num = 0  # Used by show_path() to count moves in the solution path

    def show_path(self):
        """
        Recursively output the list of moves and states along the path.

        Returns:
        - None
        """
        if self.pred is not None:
            self.pred.show_path()

        if MazeState.move_num == 0:
            print('START')
        else:
            print('Move', MazeState.move_num, 'ACTION:', self.action_from_pred)
        MazeState.move_num = MazeState.move_num + 1
        self.maze[self.pos] = MazeState.PATH

    def heuristic(self):
        """
        Calculate the minimum Euclidean distance to any exit, considering wrap-around.

        Returns:
        - Heuristic value based on the Euclidean distance to the nearest exit.
        """
        min_distance = float('inf')
        for exit_pos in self.ends:
            # Calculate distances considering wrap-around
            dx = min(abs(self.pos[0] - exit_pos[0]), self.maze.shape[0] - abs(self.pos[0] - exit_pos[0]))
            dy = min(abs(self.pos[1] - exit_pos[1]), self.maze.shape[1] - abs(self.pos[1] - exit_pos[1]))
            distance = np.linalg.norm([dx, dy])  # Euclidean distance
            min_distance = min(min_distance, distance)
        return min_distance

    def get_new_pos(self, move):
        """
        Return a new position from the current position and the specified move.

        Args:
        - move: Direction of the move (up, down, left, right).

        Returns:
        - New position after making the move.
        """
        if move == 'up':
            new_pos = (self.pos[0] - 1, self.pos[1])
        elif move == 'down':
            new_pos = (self.pos[0] + 1, self.pos[1])
        elif move == 'left':
            new_pos = (self.pos[0], self.pos[1] - 1)
        elif move == 'right':
            new_pos = (self.pos[0], self.pos[1] + 1)
        else:
            raise ('wrong direction for checking move')

        return new_pos

    def can_move(self, move):
        """
        Check if the agent can move in the given direction.

        Args:
        - move: Direction of the move (up, down, left, right).

        Returns:
        - True if the agent can move in the specified direction.
        """
        new_pos = self.get_new_pos(move)

        # Wrap around in the horizontal direction
        new_pos = (new_pos[0], new_pos[1] % self.maze.shape[1])

        # Wrap around in the vertical direction
        new_pos = (new_pos[0] % self.maze.shape[0], new_pos[1] % self.maze.shape[1])

        return self.maze[new_pos] != MazeState.WALL

    def gen_next_state(self, move):
        """
        Generate a new MazeState object by taking a move from the current state.

        Args:
        - move: Direction of the move (up, down, left, right).

        Returns:
        - New state after making the specified move.
        """
        new_pos = self.get_new_pos(move)
        if self.maze[new_pos] != MazeState.EXIT:
            self.maze[new_pos] = MazeState.VISITED
        new_state = MazeState(new_pos, self.gcost + 1, self, move)
        new_state.hcost = new_state.heuristic()
        return new_state

    def run_with_disabled_move(self, disabled_move):
        """
        Run the A* algorithm with a disabled move.

        Args:
        - disabled_move: The move to be disabled during the search.

        Returns:
        - Tuple containing a boolean indicating whether a solution was found and the number of states visited.
        """
        frontier = queue.PriorityQueue()  # Use a priority queue for efficient exploration
        closed_set = set()  # Keep track of visited states

        # Create the initial state
        start_state = MazeState(self.start)  # Initialize heuristic cost to 0
        start_state.hcost = start_state.heuristic()

        frontier.put(start_state)  # Add to priority queue

        num_states = 0
        while not frontier.empty():
            current_state = frontier.get()  # Extract state from priority queue
            num_states = num_states + 1

            if current_state.is_goal():  # Check for goal reached
                current_state.show_path()  # Return the path to the exit
                return True, num_states

            closed_set.add(current_state)

            # Generate possible next states, excluding the disabled move
            possible_moves = ['left', 'right', 'down', 'up']
            possible_moves.remove(disabled_move)  # Exclude the disabled move

            for move in possible_moves:
                if current_state.can_move(move):
                    new_state = current_state.gen_next_state(move)
                    new_state.hcost = new_state.heuristic()
                    if new_state in closed_set:
                        continue
                    if new_state not in frontier.queue:
                        frontier.put(new_state)
                    else:
                        if new_state.gcost < frontier.queue[frontier.queue.index(new_state)].gcost:
                            frontier.queue[frontier.queue.index(new_state)] = new_state
                            frontier.queue[frontier.queue.index(new_state)].hcost = new_state.heuristic()
                            heapify(frontier.queue)
        return False, num_states


def main():
    print(' Artificial Intelligence')
    print('MP1: Robot navigation')
    print('SEMESTER: Spring 2024')
    print('NAME: Bhanuprakash Banda')
    print()

    print('INITIAL MAZE')
    start_state = MazeState()
    print(start_state)

    best_move = None
    shortest_path_length = float('inf')

    for disabled_move in ['left', 'right', 'down', 'up']:
        print(f"SOLUTION AFTER DISABLED MOVE: {disabled_move}")
        solution_path, num_states = start_state.run_with_disabled_move(disabled_move)
        print(start_state)
        if solution_path:
            path_length = MazeState.move_num - 1
            print(f"\nNumber of states visited = {num_states}")
            print()
            print(f"Length of shortest path = {path_length}")
            if path_length < shortest_path_length:
                best_move = disabled_move
                shortest_path_length = path_length
        else:
            print("No solution")
        MazeState.reset_state(start_state)

    print("BEST MOVE: disable ", best_move)
    print("SHORTEST PATH LENGTH FOR BEST MOVE:", shortest_path_length)


if __name__ == "__main__":
    main()
