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
        self.states = sorted(list(states) + ['STARK_END'])
        self.alphabet = sorted(list(alphabet) + [''])
        updated_transitions = {x: {} for x in self.states}
        for state in self.states:
            for c in self.alphabet:
                if state in transition:
                    if c in transition[state]:
                        updated_transitions[state].update({c: transition[state][c]})
                    else:
                        updated_transitions[state].update({c: 'STARK_END'})
                else:
                    updated_transitions['STARK_END'].update({c: 'STARK_END'})
        self.transition = updated_transitions
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

        trace['i'].append(i)
        trace['s'].append(current_state)
        trace['c'].append('')
        
        return trace

## GEMINI

def calculate_required_blowup(automata, trace_length):
    """
    Calculates the minimum safe Blowup Factor for a specific Automata.
    """
    # 1. Estimate Logic Complexity (Degree of Transition Poly)
    num_states = len(automata.states)
    alphabet_size = len(automata.alphabet)
    
    # Total points to interpolate
    logic_degree = num_states * alphabet_size
    
    # 2. Estimate Constraint Degree
    # The Trace polynomials have degree approx 'trace_length'
    # The composition multiplies them.
    total_degree = logic_degree * trace_length
    
    # 3. Required Domain Size
    # Must be the next power of 2 strictly greater than total_degree
    req_domain_size = 2 ** math.ceil(math.log2(total_degree + 1))
    
    # 4. Calculate Blowup
    # Domain = Trace * Blowup  =>  Blowup = Domain / Trace
    # (Trace length is usually padded to a power of 2, let's assume raw length here)
    padded_trace_len = 2 ** math.ceil(math.log2(trace_length))
    
    min_blowup = req_domain_size / padded_trace_len
    
    print(f"--- Automata Complexity Analysis ---")
    print(f"States: {num_states}, Alphabet: {alphabet_size} -> Logic Deg: {logic_degree}")
    print(f"Trace Len: {padded_trace_len} -> Total Deg: {total_degree}")
    print(f"Required Domain: {req_domain_size}")
    print(f"Minimum Blowup: {min_blowup}x")
    
    return int(min_blowup)

# Example Usage:
# import json
# with open('automata.json', 'r') as f:
#    data = json.load(f)
# safe_blowup = calculate_required_blowup(data, len("I think STARKS are fun to learn"))

