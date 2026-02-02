import json
import math

# Source - https://stackoverflow.com/a/3035188
# Posted by Robert William Hanks, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-02, License - CC BY-SA 4.0

def primes(n):
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3,n,2) if sieve[i]]

# ===================================================

class Automata:
    def __init__(self, states, alphabet, transition, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_states = accept_states
        n_needed = len(self.states) + len(self.alphabet)
        safe_bound = int(n_needed * (math.log(n_needed) + 4)) + 20 if n_needed > 2 else 20
        p_list = primes(safe_bound)[:n_needed]

        # 2. Construct numeric maps
        self.state_map = {state: p_list[i] for i, state in enumerate(self.states)}
        self.caracter_map = {char: p_list[len(self.states) + i] for i, char in enumerate(self.alphabet)}

    @classmethod
    def from_json(cls, json_string):
        """Creates an instance of Automata from a JSON string."""
        data = json.loads(json_string)
        return cls(
            states=set(data["states"]),
            alphabet=set(data["alphabet"]),
            transition=data["transition"],
            start_state=data["start_state"],
            accept_states=set(data["accept_states"])
        )

    def to_json(self):
        """Serializes the automata instance into a JSON string."""
        data = {
            "states": sorted(list(self.states)),
            "alphabet": sorted(list(self.alphabet)),
            "transition": self.transition,
            "start_state": self.start_state,
            "accept_states": sorted(list(self.accept_states))
        }
        return json.dumps(data)

    def generate_trace(self, input_string):
      trace = {'i':[], 's': [], 'c': []}
      current_state = self.start_state
      i = 0
      for symbol in input_string:
          if symbol not in self.alphabet:
              return False
          trace['i'].append(i)
          i+=1
          trace['s'].append(current_state)
          trace['c'].append(symbol)
          current_state = self.transition[current_state][symbol]
    
      return trace

