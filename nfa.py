import json
import math

class Automata:
    def __init__(self, states, alphabet, transition, start_state, accept_states, groups):
        self.states = sorted(list(states))
        self.groups = groups
        self.alphabet = sorted(list(alphabet) + ['']) # '' represents Epsilon
        
        # 1. Pre-process Groups for O(1) lookup
        self.state_starts_groups = {s: [] for s in self.states}
        self.state_ends_groups = {s: [] for s in self.states}
        
        for i, group in enumerate(self.groups):
            if group["start_state"] in self.state_starts_groups:
                self.state_starts_groups[group["start_state"]].append(i)
            if group["end_state"] in self.state_ends_groups:
                self.state_ends_groups[group["end_state"]].append(i)

        # 2. Construct NFA Transitions
        updated_transitions = {x: {} for x in self.states}
        for state in self.states:
            for c in self.alphabet:
                updated_transitions[state][c] = []
                if state in transition and c in transition[state]:
                    dests = transition[state][c]
                    if isinstance(dests, list):
                        updated_transitions[state][c].extend(dests)
                    else:
                        updated_transitions[state][c].append(dests)
                        
        self.transition = updated_transitions
        self.start_states = start_state if isinstance(start_state, list) else [start_state]
        self.accept_states = set(accept_states)

        # 3. Numeric maps
        n_needed = len(self.states) + len(self.alphabet)
        def primes(n):
            out = list()
            sieve = [True] * (n+1)
            for p in range(2, n+1):
                if (sieve[p]):
                    out.append(p)
                    for i in range(p, n+1, p):
                        sieve[i] = False
            return out
            
        safe_bound = int(n_needed * (math.log(n_needed) + 4)) + 20 if n_needed > 2 else 20
        p_list = primes(safe_bound)[:n_needed]

        self.state_map = {state: p_list[i] for i, state in enumerate(self.states)}
        self.caracter_map = {char: p_list[len(self.states) + i] for i, char in enumerate(self.alphabet)}

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        return cls(
            states=set(data["states"]),
            alphabet=set(data["alphabet"]),
            groups=data["groups"],
            transition=data["transition"],
            start_state=data["start_state"],
            accept_states=set(data["accept_states"])
        )

    def to_json(self):
        data = {
            "states": sorted(list(self.states)),
            "alphabet": sorted(list(self.alphabet)),
            "groups": self.groups,
            "transition": self.transition,
            "start_state": self.start_states,
            "accept_states": sorted(list(self.accept_states))
        }
        return json.dumps(data)

    def compute(self, input_string):
        stack = []
        for s in self.start_states:
            stack.append({
                "i": 0, 
                "state": s, 
                "memory": {
                    "ACTIVE_GROUPS": {}, 
                    "GROUP_MATCHES": {}, 
                    "EPSILON_VISITED": []
                }
            })

        while len(stack) > 0:
            current = stack.pop()
            state_name = current["state"]
            i = current["i"]
            memory = current["memory"]

            if state_name in self.state_starts_groups:
                for group_idx in self.state_starts_groups[state_name]:
                    memory["ACTIVE_GROUPS"][group_idx] = i
            
            if state_name in self.state_ends_groups:
                for group_idx in self.state_ends_groups[state_name]:
                    start_pos = memory["ACTIVE_GROUPS"].get(group_idx)
                    memory["GROUP_MATCHES"][group_idx] = (group_idx, start_pos, i)

            if state_name in self.accept_states and i == len(input_string):
                return True

            def clone_memory(mem, reset_epsilon=False):
                return {
                    "ACTIVE_GROUPS": mem["ACTIVE_GROUPS"].copy(),
                    "GROUP_MATCHES": mem["GROUP_MATCHES"].copy(),
                    "EPSILON_VISITED": [] if reset_epsilon else mem["EPSILON_VISITED"].copy()
                }

            current_transitions = self.transition.get(state_name, {})

            if '' in current_transitions:
                for next_state in current_transitions['']:
                    if next_state not in memory["EPSILON_VISITED"]:
                        new_memory = clone_memory(memory, reset_epsilon=False)
                        new_memory["EPSILON_VISITED"].append(state_name)
                        stack.append({"i": i, "state": next_state, "memory": new_memory})

            if i < len(input_string):
                char = input_string[i]
                if char in current_transitions:
                    for next_state in current_transitions[char]:
                        new_memory = clone_memory(memory, reset_epsilon=True)
                        stack.append({"i": i + 1, "state": next_state, "memory": new_memory})
        return False

    def generate_trace(self, input_string):
        """
        Returns a dictionary containing lists for 'i', 's' (state), 'c' (char), 
        'started_groups', and 'ended_groups'.
        """
        stack = []
        # Trace structure now includes group columns
        empty_trace = {'i':[], 's': [], 'c': [], 'started_groups': [], 'ended_groups': []}
        
        for s in self.start_states:
            stack.append({
                "i": 0, 
                "state": s, 
                "trace": empty_trace,
                "memory": {"EPSILON_VISITED": []}
            })

        while len(stack) > 0:
            current = stack.pop()
            state_name = current["state"]
            i = current["i"]
            trace = current["trace"]
            memory = current["memory"]
            
            # Helper to append current state info to a trace
            def append_step(tr, st_name, idx, char_consumed):
                tr['i'].append(idx)
                tr['s'].append(st_name)
                tr['c'].append(char_consumed)
                # Look up groups for this state
                tr['started_groups'].append(self.state_starts_groups.get(st_name, []))
                tr['ended_groups'].append(self.state_ends_groups.get(st_name, []))

            # Check Success
            if state_name in self.accept_states and i == len(input_string):
                # Valid path found, append the final state info and return
                append_step(trace, state_name, i, '')
                return trace

            current_transitions = self.transition.get(state_name, {})
            
            # Epsilon Transitions
            if '' in current_transitions:
                for next_state in current_transitions['']:
                    if next_state not in memory["EPSILON_VISITED"]:
                        # Deep copy the trace lists
                        new_trace = {k: v[:] for k, v in trace.items()}
                        append_step(new_trace, state_name, i, '') # Record current state
                        
                        new_mem = {"EPSILON_VISITED": memory["EPSILON_VISITED"][:] + [state_name]}
                        
                        stack.append({
                            "i": i,
                            "state": next_state,
                            "trace": new_trace,
                            "memory": new_mem
                        })

            # Character Transitions
            if i < len(input_string):
                char = input_string[i]
                if char in current_transitions:
                    for next_state in current_transitions[char]:
                        new_trace = {k: v[:] for k, v in trace.items()}
                        append_step(new_trace, state_name, i, char) # Record current state
                        
                        new_mem = {"EPSILON_VISITED": []}
                        
                        stack.append({
                            "i": i + 1,
                            "state": next_state,
                            "trace": new_trace,
                            "memory": new_mem
                        })
                        
        return False