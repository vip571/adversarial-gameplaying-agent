
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        action = self.mcts(state)
        self.queue.put(action)
    
    
    def mcts(self, gameState):
        import random
        if gameState.ply_count < 2:
            return random.choice(gameState.actions())
        else:
            ###### iterative deepening ######
            depth_limit = 5
            best_move = None
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta(gameState, depth)
            return best_move
            
    ###################################################
    ###########  alpha beta pruning ###################
    def alpha_beta(self, gameState, depth):

        def min_value(gameState, alpha, beta, depth):
            if gameState.terminal_test(): 
                return gameState.utility(self.player_id)
            
            if depth <= 0: 
                return self.score(gameState)
            
            value = float("inf")
            for action in gameState.actions():
                value = min(value, max_value(gameState.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                else:
                    beta = min(beta, value)
                    
            return value

        def max_value(gameState, alpha, beta, depth):
            if gameState.terminal_test(): 
                return gameState.utility(self.player_id)
            
            if depth <= 0: 
                return self.score(gameState)
            
            value = float("-inf")
            for action in gameState.actions():
                value = max(value, min_value(gameState.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                else:
                    alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta  = float("+inf")
        best_score = float("-inf")
        best_move = None
        for action in gameState.actions():
            value = min_value(gameState.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_score:
                best_score = value
                best_move = action
        return best_move

    def score(self, gameState):
        own_loc = gameState.locs[self.player_id]
        opp_loc = gameState.locs[1 - self.player_id]
        own_liberties = gameState.liberties(own_loc)
        opp_liberties = gameState.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
class HeuristicPlayer(CustomPlayer):
    def score(self, gameState):
        own_loc = gameState.locs[self.player_id]
        opp_loc = gameState.locs[1 - self.player_id]
        own_liberties = gameState.liberties(own_loc)
        opp_liberties = gameState.liberties(opp_loc)
        return len(own_liberties) - 2 * len(opp_liberties)
